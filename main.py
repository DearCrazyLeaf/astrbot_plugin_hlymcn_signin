from __future__ import annotations

from typing import Any, Iterable
from datetime import datetime
from io import BytesIO
import base64
from pathlib import Path
from functools import partial
import html
import random
import asyncio
import a2s
import httpx
from types import SimpleNamespace
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register, StarTools
import astrbot.api.message_components as Comp
from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
    AiocqhttpMessageEvent,
)


@register(
    "HLYM服务器工具",
    "hlymcn",
    "定制：群聊签到 + 服务器查询（A2S）+ 信息查询 + 服务器远程工具",
    "0.6.5",
)
class HlymcnSignIn(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._stats_cooldowns: dict[str, float] = {}
        self._http: httpx.AsyncClient | None = None
        self._recent_header_paths: list[str] = []

    # --- core helpers & config ---
    def _get_http_client(self) -> httpx.AsyncClient:
        if self._http is None:
            limits = httpx.Limits(max_keepalive_connections=20, max_connections=50)
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(None),
                follow_redirects=True,
                http2=True,
                limits=limits,
            )
        return self._http

    def _cfg(self, key: str, default: Any) -> Any:
        value = self.config.get(key, default)
        return default if value is None else value

    def _debug_enabled(self) -> bool:
        return bool(self._cfg("debug_log", False))

    def _debug(self, msg: str, *args: Any) -> None:
        if self._debug_enabled():
            logger.info("[hlymcn_signin] " + msg, *args)

    def _normalize_group_id(self, group_id: Any) -> str:
        if group_id is None:
            return ""
        return str(group_id).strip()

    def _extract_group_id(self, event: AstrMessageEvent) -> str:
        message_obj = getattr(event, "message_obj", None)
        group_id = getattr(message_obj, "group_id", None)
        gid = self._normalize_group_id(group_id)
        if gid:
            return gid

        session_id = getattr(message_obj, "session_id", None)
        if session_id:
            last = str(session_id).split(":")[-1]
            if last.isdigit():
                return last

        return ""

    def _allowed_groups(self) -> Iterable[str]:
        raw = self._cfg("allowed_group_ids", [])
        if not isinstance(raw, list):
            return []
        return [str(x).strip() for x in raw if str(x).strip()]

    def _stats_keywords(self) -> list[str]:
        raw = self._cfg("stats_keywords", ["我的数据", "查询数据"])
        if not isinstance(raw, list):
            return []
        return [str(x).strip() for x in raw if str(x).strip()]

    def _is_group_allowed(self, group_id: Any) -> bool:
        gid = self._normalize_group_id(group_id)
        allowed = set(self._allowed_groups())
        if not allowed:
            return False
        return gid in allowed

    def _extract_user_id(self, event: AstrMessageEvent) -> str:
        # Prefer platform-specific sender id from message object.
        message_obj = getattr(event, "message_obj", None)
        sender = getattr(message_obj, "sender", None)
        for attr in ("user_id", "qq", "id"):
            value = getattr(sender, attr, None) if sender is not None else None
            if value:
                return str(value)

        getter = getattr(event, "get_sender_id", None)
        if callable(getter):
            value = getter()
            if value:
                return str(value)

        return ""

    def _is_strict_single_plain(self, event: AstrMessageEvent) -> bool:
        strict = bool(self._cfg("strict_single_plain", True))
        if not strict:
            return True

        message_obj = getattr(event, "message_obj", None)
        chain = getattr(message_obj, "message", None)
        if chain is None:
            return True
        if not isinstance(chain, list):
            return True
        if not chain:
            return False

        def _is_plain(comp: Any) -> bool:
            comp_type = getattr(comp, "type", None)
            if comp_type is not None:
                lowered = str(comp_type).lower()
                return "plain" in lowered or "text" in lowered
            return comp.__class__.__name__.lower() in {"plain", "text"}

        if self._debug_enabled():
            types = []
            for comp in chain:
                comp_type = getattr(comp, "type", None)
                types.append(str(comp_type) if comp_type is not None else comp.__class__.__name__)
            self._debug("message chain types=%s", types)

        return all(_is_plain(comp) for comp in chain)

    def _build_url(self) -> str:
        base = str(self._cfg("base_url", "https://example.com")).strip().rstrip("/")
        path = str(self._cfg("signin_path", "/api/v1/signinmobile")).strip()
        if not path.startswith("/"):
            path = "/" + path
        return f"{base}{path}"

    def _build_stats_url(self, qq_id: str) -> str:
        base = str(self._cfg("base_url", "https://example.com")).strip().rstrip("/")
        path = str(self._cfg("stats_path", "/api/v1/qq/stats")).strip()
        if not path.startswith("/"):
            path = "/" + path
        return f"{base}{path}/{qq_id}"

    def _server_status_api_base_url(self) -> str:
        base = str(self._cfg("server_status_api_base_url", "")).strip().rstrip("/")
        if not base:
            base = str(self._cfg("base_url", "")).strip().rstrip("/")
        return base

    def _server_status_api_path(self) -> str:
        path = str(self._cfg("server_status_api_path", "/api/v1/cs2/servers/batch")).strip()
        if not path.startswith("/"):
            path = "/" + path
        return path

    def _server_status_api_timeout(self, default_ms: int) -> float:
        timeout_ms = int(self._cfg("server_status_api_timeout_ms", default_ms))
        return max(timeout_ms, 1000) / 1000.0

    def _server_status_api_fallback(self) -> bool:
        return bool(self._cfg("server_status_api_fallback", True))

    def _rcon_api_base_url(self) -> str:
        return str(self._cfg("rcon_api_base_url", "")).strip().rstrip("/")

    def _rcon_password(self) -> str:
        return str(self._cfg("rcon_password", "")).strip()

    def _rcon_prefix(self) -> str:
        return str(self._cfg("rcon_command_prefix", "rcon")).strip()

    def _rcon_timeout(self) -> float:
        timeout_ms = int(self._cfg("rcon_timeout_ms", 5000))
        return max(timeout_ms, 1000) / 1000.0

    # --- parsing & formatting ---
    def _parse_number(self, value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value)
        digits = "".join(ch for ch in text if ch.isdigit() or ch == ".")
        if not digits:
            return default
        try:
            return float(digits)
        except Exception as exc:
            self._debug("parse number failed: value=%s error=%s", value, exc)
            return default

    def _parse_players_field(self, value: Any) -> tuple[int, int]:
        if value is None:
            return 0, 0
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                return int(value[0]), int(value[1])
            except Exception as exc:
                self._debug("parse players field failed: value=%s error=%s", value, exc)
                return 0, 0
        text = str(value).strip()
        if "/" in text:
            left, right = text.split("/", 1)
            try:
                return int(left.strip()), int(right.strip())
            except Exception as exc:
                self._debug("parse players field failed: value=%s error=%s", value, exc)
                return 0, 0
        try:
            return int(text), 0
        except Exception as exc:
            self._debug("parse players field failed: value=%s error=%s", value, exc)
            return 0, 0

    async def _fetch_server_status_api(self, host: str, port: int, timeout: float) -> tuple[Any, list[Any]] | None:
        base = self._server_status_api_base_url()
        if not base:
            return None

        path = self._server_status_api_path()
        url = f"{base}{path}"
        params = {"servers": f"{host}:{port}"}
        client = self._get_http_client()

        try:
            resp = await client.get(url, params=params, timeout=timeout)
        except Exception as exc:
            self._debug("server status api request failed: %s", exc)
            return None

        if resp.status_code != 200:
            self._debug("server status api status=%s", resp.status_code)
            return None

        try:
            payload = resp.json()
        except Exception as exc:
            self._debug("server status api invalid json: %s", exc)
            return None

        servers = payload.get("servers") if isinstance(payload, dict) else None
        if not isinstance(servers, list) or not servers:
            return None

        server = servers[0] if isinstance(servers[0], dict) else None
        if not server:
            return None

        raw = server.get("raw") or server.get("raw_data") or {}
        query = server.get("query") or {}

        player_count = int(self._parse_number(raw.get("playerCount"), 0))
        max_players = int(self._parse_number(raw.get("maxPlayers"), 0))
        bot_count = int(self._parse_number(raw.get("botCount"), 0))
        ping_ms = self._parse_number(query.get("ping"), 0.0)

        players_list = server.get("players") or []
        bots_list = server.get("bots") or []

        if ping_ms <= 0 and max_players == 0 and not players_list and not bots_list:
            return None

        players: list[Any] = []

        def _add_list(items: Any) -> None:
            if not isinstance(items, list):
                return
            for item in items:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "Unknown")
                score = int(self._parse_number(item.get("score"), 0))
                duration = int(self._parse_number(item.get("time"), 0))
                players.append(SimpleNamespace(name=name, score=score, duration=duration))

        _add_list(players_list)
        _add_list(bots_list)

        if not bot_count and isinstance(bots_list, list):
            bot_count = len(bots_list)

        if player_count == 0:
            player_count = len(players_list) + bot_count

        if max_players == 0 and player_count > 0:
            max_players = player_count

        info = SimpleNamespace(
            server_name=str(server.get("name") or ""),
            map_name=str(server.get("map") or ""),
            player_count=player_count,
            max_players=max_players,
            ping=ping_ms / 1000.0,
            game=str(raw.get("game") or "Counter-Strike 2"),
        )

        return info, players

    async def _handle_rcon_command(self, event: AstrMessageEvent, name: str, addr: str, command: str):
        base = self._rcon_api_base_url()
        password = self._rcon_password()
        if not base or not password:
            yield event.plain_result("RCON 未配置，请设置 rcon_api_base_url 与 rcon_password")
            return

        try:
            host, port = self._parse_address(addr)
        except ValueError as exc:
            yield event.plain_result(f"服务器地址错误：{exc}")
            return

        payload = {
            "ip": f"{host}:{port}",
            "cmd": command,
            "passwd": password,
        }
        headers = {"Content-Type": "application/json;charset=UTF-8"}
        client = self._get_http_client()
        try:
            resp = await client.post(f"{base}/rcon", json=payload, headers=headers, timeout=self._rcon_timeout())
        except Exception as exc:
            logger.error("rcon request failed: %s", exc)
            yield event.plain_result("RCON 请求失败，请稍后重试")
            return

        if resp.status_code != 200:
            yield event.plain_result(f"RCON 请求失败：HTTP {resp.status_code}")
            return

        try:
            data = resp.json()
        except Exception:
            yield event.plain_result("RCON 返回解析失败")
            return

        if isinstance(data, dict) and data.get("success", False):
            message = data.get("data") or data.get("msg") or "RCON 执行完成"
            yield event.plain_result(str(message))
            return

        message = None
        if isinstance(data, dict):
            message = data.get("msg") or data.get("message") or data.get("data")
        yield event.plain_result(str(message or "RCON 执行完成（无返回）"))

    def _server_map(self) -> dict[str, str]:
        raw = self._cfg("servers", {})
        servers: dict[str, str] = {}

        if isinstance(raw, dict):
            for key, value in raw.items():
                k = str(key).strip()
                v = str(value).strip()
                if k and v:
                    servers[k] = v
        elif isinstance(raw, list):
            for item in raw:
                text = str(item).strip()
                if not text or "=" not in text:
                    continue
                name, addr = text.split("=", 1)
                name = name.strip()
                addr = addr.strip()
                if name and addr:
                    servers[name] = addr

        return servers

    def _match_server_alias(self, text: str) -> tuple[str, str] | None:
        servers = self._server_map()
        if not servers:
            return None

        key = text.strip()
        if key in servers:
            return key, servers[key]

        lowered = key.lower()
        for name, addr in servers.items():
            if name.lower() == lowered:
                return name, addr

        return None

    def _parse_address(self, addr: str) -> tuple[str, int]:
        if ":" in addr:
            host, port_str = addr.rsplit(":", 1)
            host = host.strip()
            if not port_str.isdigit():
                raise ValueError("端口必须是数字")
            return host, int(port_str)
        return addr.strip(), 27015

    def _is_http_url(self, text: str) -> bool:
        return text.startswith("http://") or text.startswith("https://")

    def _extract_image_ref(self, result: Any) -> str | None:
        if isinstance(result, str):
            text = result.strip()
            if self._is_http_url(text):
                return text
            if text.startswith("file://"):
                return text
            path = Path(text)
            return text if path.exists() else None
        if isinstance(result, dict):
            for key in ("url", "file", "image_url", "path"):
                value = result.get(key)
                if isinstance(value, str):
                    extracted = self._extract_image_ref(value)
                    if extracted:
                        return extracted
        if isinstance(result, (list, tuple)) and result:
            for item in result:
                image_ref = self._extract_image_ref(item)
                if image_ref:
                    return image_ref
        return None

    # --- header cache ---
    def _header_cache_dir(self) -> Path:
        path = StarTools.get_data_dir() / "header_cache"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _get_header_urls(self) -> list[str]:
        raw = self._cfg("header_image_url", "")
        if isinstance(raw, list):
            return [str(x).strip() for x in raw if str(x).strip()]
        text = str(raw or "").strip()
        if not text:
            return []
        parts = []
        for token in text.replace("|", "\n").replace(";", "\n").replace(",", "\n").splitlines():
            t = token.strip()
            if t:
                parts.append(t)
        return parts

    def _list_header_cache(self) -> list[Path]:
        path = self._header_cache_dir()
        items = [p for p in path.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        return items

    def _trim_header_cache(self, max_keep: int) -> None:
        items = self._list_header_cache()
        if len(items) <= max_keep:
            return
        random.shuffle(items)
        for p in items[max_keep:]:
            try:
                p.unlink()
            except Exception:
                pass

    async def _download_header_bytes(self, url: str) -> bytes | None:
        if not url or not self._is_http_url(url):
            return None
        try:
            async with httpx.AsyncClient(
                timeout=8,
                follow_redirects=True,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                },
            ) as client:
                resp = await client.get(url)
                if resp.status_code != 200:
                    self._debug("header download status=%s url=%s", resp.status_code, url)
                    return None
                if not resp.content:
                    self._debug("header download empty content url=%s", url)
                    return None
                return resp.content
        except Exception as exc:
            self._debug("header download failed: %s", exc)
        return None

    def _save_header_cache(self, content: bytes) -> Path | None:
        try:
            img = Image.open(BytesIO(content)).convert("RGB")
        except Exception:
            self._debug("header cache save failed: invalid image bytes")
            return None
        filename = f"header_{int(datetime.now().timestamp())}_{random.randint(1000,9999)}.jpg"
        path = self._header_cache_dir() / filename
        try:
            img.save(path, format="JPEG", quality=90, optimize=True)
            return path
        except Exception:
            return None

    async def _prefetch_headers(self, count: int, max_keep: int) -> int:
        urls = self._get_header_urls()
        if not urls or count <= 0:
            return 0
        success = 0
        for _ in range(count):
            url = random.choice(urls)
            content = await self._download_header_bytes(url)
            if not content:
                continue
            saved = self._save_header_cache(content)
            if saved:
                success += 1
                self._trim_header_cache(max_keep)
        return success

    async def _silent_refresh_header_cache(self) -> None:
        urls = self._get_header_urls()
        if not urls:
            return
        max_keep = int(self._cfg("header_cache_limit", 30))
        if len(self._list_header_cache()) >= max_keep:
            return
        await self._prefetch_headers(1, max_keep)

    def _crop_cover(self, img: Image.Image, width: int, height: int) -> Image.Image:
        iw, ih = img.size
        scale = max(width / iw, height / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        img = img.resize((nw, nh), Image.LANCZOS)
        left = max(0, (nw - width) // 2)
        top = max(0, (nh - height) // 2)
        return img.crop((left, top, left + width, top + height))

    async def _get_header_image(self, width: int, height: int) -> Image.Image | None:
        cache_items = self._list_header_cache()
        if cache_items:
            recent = set(self._recent_header_paths)
            candidates = [p for p in cache_items if str(p) not in recent]
            if not candidates:
                candidates = cache_items
            random.shuffle(candidates)
            for p in candidates:
                try:
                    img = Image.open(p).convert("RGB")
                    self._recent_header_paths.append(str(p))
                    self._recent_header_paths = self._recent_header_paths[-3:]
                    return self._crop_cover(img, width, height)
                except Exception:
                    try:
                        p.unlink()
                    except Exception:
                        pass

        urls = self._get_header_urls()
        if not urls:
            return None
        url = random.choice(urls)
        content = await self._download_header_bytes(url)
        if not content:
            return None
        saved = self._save_header_cache(content)
        if saved and saved.exists():
            try:
                img = Image.open(saved).convert("RGB")
                self._recent_header_paths.append(str(saved))
                self._recent_header_paths = self._recent_header_paths[-3:]
                return self._crop_cover(img, width, height)
            except Exception:
                return None
        try:
            img = Image.open(BytesIO(content)).convert("RGB")
            return self._crop_cover(img, width, height)
        except Exception:
            return None

    # --- render helpers ---
    def _load_font(self, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        font_dir = Path(__file__).with_name("font")
        font_files: list[Path] = []
        if font_dir.exists():
            font_files = sorted(font_dir.glob("*.ttf")) + sorted(font_dir.glob("*.otf"))

        candidates: list[Path] = []
        if font_files:
            if bold:
                for f in font_files:
                    name = f.name.lower()
                    if "bd" in name or "bold" in name:
                        candidates.append(f)
                candidates.append(font_files[0])
            else:
                candidates.append(font_files[0])

        for path in candidates:
            try:
                return ImageFont.truetype(str(path), size=size)
            except Exception as exc:
                self._debug("load font failed: %s (%s)", path, exc)

        self._debug("font not found in plugin font directory, using default")
        return ImageFont.load_default()

    def _wrap_text(self, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
        if not text:
            return [""]
        lines: list[str] = []
        current = ""
        for ch in text:
            trial = current + ch
            width = font.getbbox(trial)[2]
            if width <= max_width:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = ch
        if current:
            lines.append(current)
        return lines

    def _build_players_rows(
        self,
        players: list[Any],
        max_players_show: int,
    ) -> list[tuple[int, str, int, str]]:
        human_players, _ = self._split_players(players)
        rows: list[tuple[int, str, int, str]] = []
        if human_players:
            players_sorted = sorted(human_players, key=lambda p: getattr(p, "score", 0), reverse=True)
            for idx, p in enumerate(players_sorted[:max_players_show], 1):
                name = (p.name or "玩家").strip() if hasattr(p, "name") else "玩家"
                score = int(getattr(p, "score", 0))
                duration = int(getattr(p, "duration", 0))
                minutes = duration // 60
                seconds = duration % 60
                rows.append((idx, name, score, f"{minutes:02d}:{seconds:02d}"))
        else:
            rows.append((1, "暂无玩家", 0, ""))
        return rows

    def _calc_info_layout(
        self,
        info_items: list[tuple[str, str, str]],
        label_font: ImageFont.FreeTypeFont,
        value_font: ImageFont.FreeTypeFont,
        card_width: int,
        padding: int,
        col_gap: int,
        label_min_w: int,
        value_gap: int,
    ) -> SimpleNamespace:
        col_width = (card_width - padding * 2 - col_gap) // 2
        left_items = info_items[:3]
        right_items = info_items[3:]
        row_count = max(len(left_items), len(right_items))

        left_label_w = label_min_w
        right_label_w = label_min_w
        for icon, label, _ in left_items:
            label_text = f"{icon} {label}".strip()
            left_label_w = max(left_label_w, label_font.getbbox(label_text)[2])
        for icon, label, _ in right_items:
            label_text = f"{icon} {label}".strip()
            right_label_w = max(right_label_w, label_font.getbbox(label_text)[2])

        value_w_left = max(120, col_width - left_label_w - value_gap)
        value_w_right = max(120, col_width - right_label_w - value_gap)

        info_lines_count = 0
        for i in range(row_count):
            left_value = left_items[i][2] if i < len(left_items) else ""
            right_value = right_items[i][2] if i < len(right_items) else ""
            left_lines = self._wrap_text(str(left_value), value_font, value_w_left)
            right_lines = self._wrap_text(str(right_value), value_font, value_w_right)
            info_lines_count += max(1, len(left_lines), len(right_lines))

        return SimpleNamespace(
            col_width=col_width,
            left_items=left_items,
            right_items=right_items,
            row_count=row_count,
            left_label_w=left_label_w,
            right_label_w=right_label_w,
            value_w_left=value_w_left,
            value_w_right=value_w_right,
            info_lines_count=info_lines_count,
        )

    def _draw_info_section(
        self,
        draw: ImageDraw.ImageDraw,
        y: int,
        layout: SimpleNamespace,
        padding: int,
        col_gap: int,
        label_font: ImageFont.FreeTypeFont,
        value_font: ImageFont.FreeTypeFont,
        label_color: tuple[int, int, int],
        text_color: tuple[int, int, int],
        line_height: int,
        value_gap: int,
        ping_label: str,
        game_label: str,
        ping_ms: float,
    ) -> int:
        left_items = layout.left_items
        right_items = layout.right_items
        row_count = layout.row_count
        left_label_w = layout.left_label_w
        right_label_w = layout.right_label_w
        value_w_left = layout.value_w_left
        value_w_right = layout.value_w_right

        left_x = padding
        right_x = padding + layout.col_width + col_gap

        def _first_line_width(value: str, max_w: int) -> int:
            lines = self._wrap_text(str(value), value_font, max_w)
            if not lines:
                return 0
            return value_font.getbbox(lines[0])[2]

        right_ping_value = ""
        right_game_value = ""
        for _, label, value in right_items:
            if label == ping_label:
                right_ping_value = str(value)
            if label == game_label:
                right_game_value = str(value)

        max_bar_cap = _first_line_width(right_game_value, value_w_right) if right_game_value else 120

        for i in range(row_count):
            _, left_label, left_value = left_items[i] if i < len(left_items) else ("", "", "")
            _, right_label, right_value = right_items[i] if i < len(right_items) else ("", "", "")

            left_label_text = left_label.strip()
            right_label_text = right_label.strip()

            left_lines = self._wrap_text(str(left_value), value_font, value_w_left)
            right_lines = self._wrap_text(str(right_value), value_font, value_w_right)
            row_lines = max(1, len(left_lines), len(right_lines))

            if left_label_text:
                draw.text((left_x, y), left_label_text, font=label_font, fill=label_color)
            if right_label_text:
                draw.text((right_x, y), right_label_text, font=label_font, fill=label_color)

            for idx, line in enumerate(left_lines):
                draw.text(
                    (left_x + left_label_w + value_gap, y + idx * line_height),
                    line,
                    font=value_font,
                    fill=text_color,
                )
            for idx, line in enumerate(right_lines):
                draw.text(
                    (right_x + right_label_w + value_gap, y + idx * line_height),
                    line,
                    font=value_font,
                    fill=text_color,
                )
                if right_label == ping_label and idx == 0:
                    text_w = value_font.getbbox(line)[2]
                    bar_x = right_x + right_label_w + value_gap + text_w + 10
                    bar_h = 6
                    bar_y = y + idx * line_height + (line_height - bar_h) // 2
                    max_bar_w = min(max_bar_cap, value_w_right - text_w - 10)
                    bar_w = max(0, int(max_bar_w * 0.45))
                    ping_clamped = max(0, min(300.0, ping_ms))
                    ratio = ping_clamped / 300.0
                    if ratio <= 0.33:
                        bar_color = (62, 193, 86)
                    elif ratio <= 0.66:
                        bar_color = (64, 126, 255)
                    elif ratio <= 0.85:
                        bar_color = (255, 159, 64)
                    else:
                        bar_color = (220, 70, 70)
                    if bar_w > 0:
                        draw.rounded_rectangle(
                            [bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
                            radius=bar_h // 2,
                            fill=(80, 80, 80),
                        )
                        fill_w = max(6, int(bar_w * min(1.0, ratio * 1.2)))
                        draw.rounded_rectangle(
                            [bar_x, bar_y, bar_x + fill_w, bar_y + bar_h],
                            radius=bar_h // 2,
                            fill=bar_color,
                        )

            y += line_height * row_lines

        return y

    def _draw_players_section(
        self,
        card: Image.Image,
        draw: ImageDraw.ImageDraw,
        y: int,
        players_rows: list[tuple[int, str, int, str]],
        rows_count: int,
        card_width: int,
        padding: int,
        scale: float,
        small_font: ImageFont.FreeTypeFont,
        text_color: tuple[int, int, int],
        muted_color: tuple[int, int, int],
    ) -> tuple[Image.Image, ImageDraw.ImageDraw, int]:
        players_line_height = int(20 * scale)
        players_header_gap = int(8 * scale)

        table_edge = max(int(8 * scale), padding - int(12 * scale))
        table_x = table_edge
        table_right = card_width - table_edge
        col_gap = int(12 * scale)
        col_idx_w = int(36 * scale)
        col_score_w = int(70 * scale)
        col_time_w = int(90 * scale)
        col_name_w = table_right - table_x - col_idx_w - col_score_w - col_time_w - col_gap * 3
        col_name_w = max(120, col_name_w)

        x_idx = table_x
        x_name = x_idx + col_idx_w + col_gap
        x_score = x_name + col_name_w + col_gap
        x_time = x_score + col_score_w + col_gap

        draw.text((x_idx, y), "#", font=small_font, fill=muted_color)
        draw.text((x_name, y), "名称", font=small_font, fill=muted_color)
        draw.text((x_score, y), "得分", font=small_font, fill=muted_color)
        draw.text((x_time, y), "时长", font=small_font, fill=muted_color)
        y += players_line_height + players_header_gap

        row_overlay = Image.new("RGBA", (card_width, card.height), (0, 0, 0, 0))
        row_draw = ImageDraw.Draw(row_overlay)
        row_records: list[tuple[int, str, int, str, int]] = []
        for idx, name, score, playtime in players_rows:
            row_fill = (255, 255, 255, 16) if idx % 2 == 0 else (255, 255, 255, 8)
            row_outline = (255, 255, 255, 22)
            row_top = y - 1
            row_bottom = y + players_line_height - 1
            row_draw.rectangle(
                [table_x - 2, row_top, table_right + 2, row_bottom],
                fill=row_fill,
                outline=row_outline,
                width=1,
            )
            row_records.append((idx, name, score, playtime, y))
            y += players_line_height

        card = Image.alpha_composite(card, row_overlay)
        draw = ImageDraw.Draw(card)
        for idx, name, score, playtime, row_y in row_records:
            draw.text((x_idx, row_y), str(idx), font=small_font, fill=text_color)
            draw.text((x_name, row_y), name, font=small_font, fill=text_color)
            score_text = "-" if name == "暂无玩家" else str(score)
            draw.text((x_score, row_y), score_text, font=small_font, fill=text_color)
            if playtime:
                draw.text((x_time, row_y), playtime, font=small_font, fill=text_color)

        extra_rows = max(0, rows_count - len(players_rows))
        if extra_rows:
            y += players_line_height * extra_rows
        return card, draw, y

    def _draw_footer(
        self,
        draw: ImageDraw.ImageDraw,
        card_width: int,
        padding: int,
        line_color: tuple[int, int, int],
        now_text: str,
        footer_font: ImageFont.FreeTypeFont,
        footer_line_gap: int,
        bottom_padding: int,
        card_height: int,
    ) -> None:
        footer_h = footer_font.getbbox("测")[3] + 2
        footer_y = card_height - bottom_padding - footer_h
        line_y = footer_y - footer_line_gap
        draw.line((padding, line_y, card_width - padding, line_y), fill=line_color, width=1)
        draw.text((padding, footer_y), f"查询时间：{now_text}", font=footer_font, fill=(140, 145, 150))

    def _section_layout(
        self,
        items: list[tuple[str, str]],
        label_font: ImageFont.FreeTypeFont,
        value_font: ImageFont.FreeTypeFont,
        col_width: int,
        label_min_w: int,
        value_gap: int,
    ) -> SimpleNamespace:
        label_w = label_min_w
        for label, _ in items:
            label_w = max(label_w, label_font.getbbox(label)[2])
        value_w = max(120, col_width - label_w - value_gap)
        return SimpleNamespace(label_w=label_w, value_w=value_w)

    def _draw_two_col_section(
        self,
        draw: ImageDraw.ImageDraw,
        y: int,
        title: str,
        items: list[tuple[str, str]],
        col_width: int,
        label_font: ImageFont.FreeTypeFont,
        value_font: ImageFont.FreeTypeFont,
        section_font: ImageFont.FreeTypeFont,
        label_color: tuple[int, int, int],
        text_color: tuple[int, int, int],
        padding: int,
        col_gap: int,
        line_height: int,
        value_gap: int,
        layout: SimpleNamespace,
    ) -> int:
        draw.text((padding, y), title, font=section_font, fill=text_color)
        y += section_font.getbbox("测")[3] + 6

        left_items = items[::2]
        right_items = items[1::2]
        row_count = max(len(left_items), len(right_items))

        left_x = padding
        right_x = padding + col_width + col_gap
        label_w = layout.label_w
        value_w = layout.value_w

        for i in range(row_count):
            left_label, left_value = left_items[i] if i < len(left_items) else ("", "")
            right_label, right_value = right_items[i] if i < len(right_items) else ("", "")

            if left_label:
                draw.text((left_x, y), left_label, font=label_font, fill=label_color)
            if right_label:
                draw.text((right_x, y), right_label, font=label_font, fill=label_color)

            left_lines = self._wrap_text(str(left_value), value_font, value_w)
            right_lines = self._wrap_text(str(right_value), value_font, value_w)
            if left_lines:
                draw.text(
                    (left_x + label_w + value_gap, y),
                    left_lines[0],
                    font=value_font,
                    fill=text_color,
                )
            if right_lines:
                draw.text(
                    (right_x + label_w + value_gap, y),
                    right_lines[0],
                    font=value_font,
                    fill=text_color,
                )

            y += line_height

        return y

    # --- render sizing ---
    def _estimate_card_size(
        self,
        host: str,
        port: int,
        info: Any,
        players: list[Any],
        max_players_show: int,
        scale: float = 1.2,
    ) -> tuple[int, int]:
        card_width = int(820 * scale)
        padding = int(28 * scale)
        line_height = int(26 * scale)
        players_title_height = int(26 * scale)
        players_line_height = int(20 * scale)
        divider_gap = int(12 * scale)
        players_header_gap = int(8 * scale)
        title_font = self._load_font(int(26 * scale), bold=True)
        label_font = self._load_font(int(18 * scale), bold=True)
        value_font = self._load_font(int(18 * scale), bold=False)
        small_font = self._load_font(int(14 * scale), bold=False)

        ping_ms = float(getattr(info, "ping", 0) * 1000)
        _, bot_players = self._split_players(players)
        bot_count = len(bot_players)
        bot_suffix = f" ({bot_count} BOT)" if bot_count else ""
        info_items = [
            ("", "????", "example.com"),
            ("", "???IP", f"{host}:{port}"),
            ("", "?????", f"{getattr(info, 'player_count', 0)}/{getattr(info, 'max_players', 0)}{bot_suffix}"),
            ("", "????", getattr(info, "map_name", "")),
            ("", "????", f"{ping_ms:.0f} ms"),
            ("", "??", getattr(info, "game", "")),
        ]

        players_rows = self._build_players_rows(players, max_players_show)

        col_gap = int(36 * scale)
        label_min_w = int(120 * scale)
        value_gap = int(16 * scale)
        info_layout = self._calc_info_layout(
            info_items,
            label_font,
            value_font,
            card_width,
            padding,
            col_gap,
            label_min_w,
            value_gap,
        )
        info_lines_count = info_layout.info_lines_count

        title_height = title_font.getbbox("?")[3]
        title_block_height = title_height + divider_gap * 2
        info_height = info_lines_count * line_height
        max_players_cfg = int(getattr(info, "max_players", 0) or 0)
        base_rows = (max_players_cfg + 1) // 2 if max_players_cfg > 0 else max_players_show
        rows_count = max(1, len(players_rows), base_rows)
        players_bottom_padding = int(8 * scale)
        players_height = (
            players_title_height
            + players_header_gap
            + players_line_height
            + rows_count * players_line_height
            + players_bottom_padding
        )
        footer_height = small_font.getbbox("?")[3] + int(8 * scale)
        bottom_padding = int(4 * scale)
        footer_line_gap = max(1, int(divider_gap * 0.35))
        footer_block_height = footer_line_gap + footer_height + bottom_padding
        content_height = (
            title_block_height
            + info_height
            + divider_gap * 2
            + players_height
            + footer_block_height
        )
        title_area_top = max(int(16 * scale), padding - int(10 * scale))
        card_height = title_area_top + content_height
        return card_width, card_height

    def _estimate_stats_card_size(
        self,
        title: str,
        qq_id: str,
        steam_id: str,
        credits: Any,
        kd: str | None,
        winrate: str | None,
        kills: int,
        first_blood: int,
        deaths: int,
        grenades: int,
        shoots: int,
        mvp: int,
        rounds_total: int,
        round_win: int,
        round_lose: int,
        total_time: str,
        ct_time: str,
        t_time: str,
        spec_time: str,
        alive_time: str,
        dead_time: str,
    ) -> tuple[int, int]:
        card_width = 820
        padding = 28
        col_gap = 36
        col_width = (card_width - padding * 2 - col_gap) // 2
        label_min_w = 120
        value_gap = 16
        line_height = 26
        divider_gap = 12
        time_gap = divider_gap * 3
        bottom_padding = 12

        title_font = self._load_font(26, bold=True)
        section_font = self._load_font(20, bold=True)
        label_font = self._load_font(18, bold=True)
        value_font = self._load_font(18, bold=False)
        small_font = self._load_font(14, bold=False)

        title_text = title or "??????"
        account_items = [
            ("????", title_text),
            ("QQ?", qq_id),
            ("Steam64", steam_id),
            ("?????", str(credits)),
        ]
        stats_items = [
            ("K/D", kd or "??"),
            ("??", winrate or "??"),
            ("???", str(kills)),
            ("???", str(first_blood)),
            ("???", str(deaths)),
            ("???", str(grenades)),
            ("???", str(shoots)),
            ("MVP", str(mvp)),
            ("???", str(rounds_total)),
            ("?/?", f"{round_win}/{round_lose}"),
        ]
        time_items = [
            ("????", alive_time),
            ("CT??", ct_time),
            ("T??", t_time),
            ("???", total_time),
            ("????", spec_time),
            ("????", dead_time),
        ]

        title_height = title_font.getbbox("?")[3]
        section_title_height = section_font.getbbox("?")[3]
        footer_height = small_font.getbbox("?")[3] + 8

        def _section_height(items: list[tuple[str, str]]) -> int:
            rows = (len(items) + 1) // 2
            return section_title_height + 6 + rows * line_height

        content_height = (
            title_height
            + 8
            + divider_gap
            + _section_height(account_items)
            + divider_gap * 2 + 2
            + _section_height(stats_items)
            + divider_gap * 2 + 2
            + _section_height(time_items)
            + divider_gap
            + time_gap
            + footer_height
            + bottom_padding
        )
        card_height = padding * 2 + content_height
        return card_width, card_height

    # --- server status card rendering ---
    async def _render_card_pil(
        self,
        host: str,
        port: int,
        server_alias: str,
        info: Any,
        players: list[Any],
        now_text: str,
        max_players_show: int,
    ) -> bytes | None:
        card_width, card_height = self._estimate_card_size(host, port, info, players, max_players_show)
        header_image = await self._get_header_image(card_width, card_height)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            partial(
                self._render_card_pil_sync,
                host,
                port,
                server_alias,
                info,
                players,
                now_text,
                max_players_show,
                header_image,
            ),
        )

    def _render_card_pil_sync(
        self,
        host: str,
        port: int,
        server_alias: str,
        info: Any,
        players: list[Any],
        now_text: str,
        max_players_show: int,
        header_image: Image.Image | None,
    ) -> bytes | None:
        scale = 1.2
        card_width = int(820 * scale)
        header_h = 0
        padding = int(28 * scale)
        outer_margin = 0
        corner_radius = 0
        bg_color = (44, 44, 44)
        card_color = (58, 58, 58)
        line_color = (75, 75, 75)
        text_color = (245, 245, 245)
        muted_color = (160, 166, 173)
        label_color = (170, 176, 182)

        title_font = self._load_font(int(26 * scale), bold=True)
        section_font = self._load_font(int(20 * scale), bold=True)
        label_font = self._load_font(int(18 * scale), bold=True)
        value_font = self._load_font(int(18 * scale), bold=False)
        small_font = self._load_font(int(14 * scale), bold=False)

        ping_ms = float(getattr(info, "ping", 0) * 1000)
        _, bot_players = self._split_players(players)
        bot_count = len(bot_players)
        bot_suffix = f" ({bot_count} BOT)" if bot_count else ""
        info_items = [
            ("", "官方网站", "example.com"),
            ("", "服务器IP", f"{host}:{port}"),
            ("", "服务器人数", f"{getattr(info, 'player_count', 0)}/{getattr(info, 'max_players', 0)}{bot_suffix}"),
            ("", "当前地图", getattr(info, "map_name", "")),
            ("", "当前延迟", f"{ping_ms:.0f} ms"),
            ("", "游戏", getattr(info, "game", "")),
        ]

        line_height = int(26 * scale)
        players_title_height = int(26 * scale)
        players_line_height = int(20 * scale)
        divider_gap = int(12 * scale)
        players_header_gap = int(8 * scale)
        time_gap = 0
        title_height = title_font.getbbox("测")[3]
        players_rows = self._build_players_rows(players, max_players_show)

        col_gap = int(36 * scale)
        label_min_w = int(120 * scale)
        value_gap = int(16 * scale)
        info_layout = self._calc_info_layout(
            info_items,
            label_font,
            value_font,
            card_width,
            padding,
            col_gap,
            label_min_w,
            value_gap,
        )
        info_lines_count = info_layout.info_lines_count

        title_block_height = title_height + divider_gap * 2
        info_height = info_lines_count * line_height
        max_players_cfg = int(getattr(info, "max_players", 0) or 0)
        base_rows = (max_players_cfg + 1) // 2 if max_players_cfg > 0 else max_players_show
        rows_count = max(1, len(players_rows), base_rows)
        players_bottom_padding = int(8 * scale)
        players_height = (
            players_title_height
            + players_header_gap
            + players_line_height
            + rows_count * players_line_height
            + players_bottom_padding
        )
        footer_height = small_font.getbbox("测")[3] + int(8 * scale)
        bottom_padding = int(4 * scale)
        footer_line_gap = max(1, int(divider_gap * 0.35))
        footer_block_height = footer_line_gap + footer_height + bottom_padding
        content_height = (
            title_block_height
            + info_height
            + divider_gap * 2
            + players_height
            + time_gap
            + footer_block_height
        )
        title_area_top = max(int(16 * scale), padding - int(10 * scale))
        card_height = title_area_top + content_height
        players_title_y = (
            title_area_top + title_block_height + info_height + divider_gap * 3
        )

        base = Image.new(
            "RGB",
            (card_width + outer_margin * 2, card_height + outer_margin * 2),
            bg_color,
        )
        card = Image.new("RGBA", (card_width, card_height), (0, 0, 0, 0))
        card_draw = ImageDraw.Draw(card)
        card_draw.rounded_rectangle(
            [0, 0, card_width, card_height],
            radius=corner_radius,
            fill=card_color,
        )

        bg = header_image
        if bg is not None and (bg.width != card_width or bg.height != card_height):
            bg = self._crop_cover(bg, card_width, card_height)
        if bg is None:
            bg = Image.new("RGB", (card_width, card_height), (60, 60, 60))
        bg_rgba = bg.convert("RGBA")
        bg_blur = bg_rgba.filter(ImageFilter.GaussianBlur(2.4))
        bg_rgba = Image.blend(bg_rgba, bg_blur, 0.42)
        overlay = Image.new("RGBA", (card_width, card_height), (0, 0, 0, 140))
        bg_rgba = Image.alpha_composite(bg_rgba, overlay)

        mask = Image.new("L", (card_width, card_height), 0)
        ImageDraw.Draw(mask).rounded_rectangle(
            [0, 0, card_width, card_height],
            radius=corner_radius,
            fill=255,
        )
        card.paste(bg_rgba, (0, 0), mask)

        # 取消面板毛玻璃，仅保留整体轻微模糊

        draw = ImageDraw.Draw(card)

        title_text = getattr(info, "server_name", "") or ""
        title_gap = divider_gap
        title_block_height = title_height + title_gap * 2
        y = title_area_top
        title_y = y + (title_block_height - title_height) // 2 - 2
        draw.text((padding, title_y), title_text, font=title_font, fill=text_color)

        # 服务器别名徽章已移除
        y += title_block_height
        draw.line((padding, y, card_width - padding, y), fill=line_color, width=2)
        y += divider_gap

        ping_label = "当前延迟"
        game_label = "游戏"
        y = self._draw_info_section(
            draw,
            y,
            info_layout,
            padding,
            col_gap,
            label_font,
            value_font,
            label_color,
            text_color,
            line_height,
            value_gap,
            ping_label,
            game_label,
            ping_ms,
        )

        y += divider_gap
        draw.line((padding, y, card_width - padding, y), fill=line_color, width=1)
        y += divider_gap

        draw.text((padding, y), "玩家列表", font=label_font, fill=text_color)
        y += players_title_height
        card, draw, y = self._draw_players_section(
            card,
            draw,
            y,
            players_rows,
            rows_count,
            card_width,
            padding,
            scale,
            small_font,
            text_color,
            muted_color,
        )
        y += players_bottom_padding

        footer_font = self._load_font(int(12 * scale), bold=False)
        self._draw_footer(
            draw,
            card_width,
            padding,
            line_color,
            now_text,
            footer_font,
            footer_line_gap,
            bottom_padding,
            card_height,
        )

        base.paste(card, (outer_margin, outer_margin), card)
        output = BytesIO()
        base.save(output, format="JPEG", quality=85, optimize=True)
        return output.getvalue()

    # --- HTML rendering (fallback) ---
    async def _try_html_render(self, html_content: str) -> tuple[str | None, str]:
        render_func = getattr(self, "html_render", None)
        if not callable(render_func):
            return None, "html_render not available"

        options = {
            "full_page": True,
            "type": "png",
            "scale": "device",
            "device_scale_factor_level": "normal",
        }

        try_orders = (
            ("legacy", lambda: render_func(html_content, {}, True, options)),
            ("keyword", lambda: render_func(html_content, {}, return_url=True, options=options)),
            ("simple", lambda: render_func(html_content, {}, options=options)),
        )

        last_error = ""
        for label, call in try_orders:
            try:
                result = await call()
                image_ref = self._extract_image_ref(result)
                if image_ref:
                    return image_ref, f"ok via {label}"
                last_error = f"{label} returned no image ref"
            except TypeError:
                last_error = f"{label} signature not supported"
                continue
            except Exception as exc:
                last_error = f"{label} error: {exc}"
                break

        return None, last_error or "unknown error"

    def _build_html_card(
        self,
        host: str,
        port: int,
        info: Any,
        players_html: str,
        header_url: str,
        now_text: str,
    ) -> str:
        server_name = html.escape(getattr(info, "server_name", "") or "")
        game = html.escape(getattr(info, "game", "") or "")
        map_name = html.escape(getattr(info, "map_name", "") or "")
        player_count = getattr(info, "player_count", 0)
        max_players = getattr(info, "max_players", 0)
        ping_ms = f"{getattr(info, 'ping', 0) * 1000:.0f}"
        host_safe = html.escape(f"{host}:{port}")

        header_html = ""
        if header_url and self._is_http_url(header_url):
            header_html = (
                "<div class=\"header\">"
                f"<img src=\"{header_url}\" alt=\"header\" />"
                "<div class=\"header-mask\"></div>"
                "</div>"
            )
        else:
            header_html = (
                "<div class=\"header no-image\">"
                "<div class=\"header-title\">服务器状态更新</div>"
                "</div>"
            )

        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <style>
    :root {{
      --card-width: 760px;
      --bg: #2c2c2c;
      --card: #3a3a3a;
      --text: #f2f2f2;
      --muted: #a0a6ad;
      --line: #4b4b4b;
      --chip: #4a4a4a;
      --chip-border: #5a5a5a;
      --accent: #4ea1ff;
    }}
    * {{
      box-sizing: border-box;
    }}
    html, body {{
      margin: 0;
      padding: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Microsoft YaHei", "Noto Sans SC", "PingFang SC", sans-serif;
    }}
    .card {{
      width: var(--card-width);
      background: var(--card);
      border-radius: 18px;
      overflow: hidden;
      box-shadow: 0 24px 60px rgba(0, 0, 0, 0.35);
    }}
    .header {{
      position: relative;
      height: 260px;
      background: #444;
    }}
    .header.no-image {{
      height: 180px;
      display: flex;
      align-items: center;
      justify-content: center;
      background: linear-gradient(120deg, #3c3c3c 0%, #2f2f2f 100%);
    }}
    .header-title {{
      font-size: 26px;
      font-weight: 700;
      letter-spacing: 1px;
    }}
    .header img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }}
    .header-mask {{
      position: absolute;
      inset: 0;
      background: linear-gradient(180deg, rgba(0,0,0,0.05) 0%, rgba(0,0,0,0.65) 100%);
    }}
    .content {{
      padding: 22px 26px 18px;
    }}
    .title {{
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 22px;
      font-weight: 700;
    }}
    .badge {{
      padding: 4px 12px;
      border-radius: 999px;
      background: var(--chip);
      border: 1px solid var(--chip-border);
      font-size: 12px;
      letter-spacing: 1px;
      text-transform: uppercase;
    }}
    .accent {{
      height: 3px;
      width: 100%;
      margin: 14px 0 16px;
      background: linear-gradient(90deg, var(--accent), transparent);
      opacity: 0.8;
      border-radius: 999px;
    }}
    .info-grid {{
      margin-top: 4px;
      display: grid;
      grid-template-columns: 96px 1fr;
      gap: 8px 12px;
    }}
    .label {{
      color: var(--muted);
      font-size: 13px;
    }}
    .value {{
      font-size: 14px;
      color: var(--text);
      word-break: break-all;
    }}
    .divider {{
      height: 1px;
      background: var(--line);
      margin: 18px 0 14px;
    }}
    .section-title {{
      font-size: 16px;
      font-weight: 600;
      margin-bottom: 10px;
    }}
    .players-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    .players-table th {{
      text-align: left;
      color: var(--muted);
      font-weight: 600;
      padding: 6px 0;
      border-bottom: 1px solid var(--line);
    }}
    .players-table td {{
      padding: 6px 0;
      border-bottom: 1px dashed #474747;
    }}
    .players-empty {{
      background: #424242;
      border-radius: 10px;
      padding: 10px 12px;
      color: #d0d0d0;
      font-size: 13px;
    }}
    .footer {{
      margin-top: 12px;
      color: var(--muted);
      font-size: 11px;
      text-align: right;
    }}
  </style>
</head>
<body>
  <div class="card">
    {header_html}
    <div class="content">
      <div class="title">
        <span class="badge">服务器状态更新</span>
      </div>
      <div class="accent"></div>
      <div class="info-grid">
        <div class="label">服务器名称</div><div class="value">{server_name}</div>
        <div class="label">官方网站</div><div class="value">example.com</div>
        <div class="label">服务器IP</div><div class="value">{host_safe}</div>
        <div class="label">服务器人数</div><div class="value">{player_count}/{max_players}</div>
        <div class="label">当前地图</div><div class="value">{map_name}</div>
        <div class="label">当前延迟</div><div class="value">{ping_ms} ms</div>
        <div class="label">游戏</div><div class="value">{game}</div>
      </div>
      <div class="divider"></div>
      <div class="section-title">玩家列表</div>
      {players_html}
      <div class="footer">查询时间：{now_text}</div>
    </div>
  </div>
</body>
</html>"""

    async def _render_html_to_image_ref(self, html_content: str) -> str | None:
        render_strategies = [
            {
                "full_page": True,
                "type": "png",
                "scale": "device",
                "device_scale_factor_level": "normal",
            },
            {
                "full_page": True,
                "type": "jpeg",
                "quality": 80,
                "scale": "device",
                "device_scale_factor_level": "low",
            },
        ]

        render_func = getattr(self, "html_render", None)
        if not callable(render_func):
            self._debug("html_render not available")
            return None

        for options in render_strategies:
            try:
                try:
                    result = await render_func(html_content, {}, True, options)
                except TypeError:
                    try:
                        result = await render_func(
                            html_content, {}, return_url=True, options=options
                        )
                    except TypeError:
                        result = await render_func(html_content, {}, options=options)

                image_ref = self._extract_image_ref(result)
                if image_ref:
                    return image_ref
            except Exception as exc:
                logger.warning("html render url failed (%s): %s", options, exc)

        return None

    async def _render_html_to_bytes(self, html_content: str) -> bytes | None:
        render_func = getattr(self, "html_render", None)
        if not callable(render_func):
            self._debug("html_render not available")
            return None

        options = {
            "full_page": True,
            "type": "jpeg",
            "quality": 85,
            "scale": "device",
        }

        try_orders = (
            ("legacy", lambda: render_func(html_content, {}, False, options)),
            ("keyword", lambda: render_func(html_content, {}, return_url=False, options=options)),
            ("simple", lambda: render_func(html_content, {}, options=options)),
        )

        last_error = ""
        for label, call in try_orders:
            try:
                result = await call()
                if isinstance(result, (bytes, bytearray)) and result:
                    return bytes(result)
                last_error = f"{label} returned no bytes"
            except TypeError:
                last_error = f"{label} signature not supported"
                continue
            except Exception as exc:
                last_error = f"{label} error: {exc}"
                break

        self._debug("html_render bytes failed: %s", last_error)
        return None

    @filter.command("prefetch_header", alias={"预载头图"})
    async def prefetch_header(self, event: AstrMessageEvent):
        """预载头图到缓存"""
        urls = self._get_header_urls()
        if not urls:
            yield event.plain_result("未配置头图链接，无法预载")
            return
        prefetch_count = int(self._cfg("header_prefetch_count", 10))
        max_keep = int(self._cfg("header_cache_limit", 30))
        success = await self._prefetch_headers(prefetch_count, max_keep)
        cache_size = len(self._list_header_cache())
        yield event.plain_result(
            f"预载完成：成功 {success}/{prefetch_count}，缓存中共 {cache_size} 张"
        )
    async def _handle_signin(self, event: AstrMessageEvent):
        user_id = self._extract_user_id(event)
        if not user_id:
            self._debug("fail: user id not found")
            yield event.plain_result("未获取到用户QQ号，无法签到")
            return

        url = self._build_url()
        params = {
            "userId": user_id,
            "deviceType": str(self._cfg("device_type", "2")),
            "signinType": str(self._cfg("signin_type", "1")),
        }
        timeout_ms = int(self._cfg("timeout_ms", 5000))
        timeout = max(timeout_ms, 1000) / 1000.0

        self._debug("request url=%s params=%s", url, params)

        resp = None
        last_exc: Exception | None = None
        client = self._get_http_client()
        signin_timeout = httpx.Timeout(connect=5.0, read=None, write=5.0, pool=5.0)
        for attempt in range(2):
            try:
                resp = await client.get(url, params=params, timeout=signin_timeout)
                break
            except (httpx.TimeoutException, httpx.RequestError) as exc:
                last_exc = exc
                if attempt < 1:
                    await asyncio.sleep(0.6)
                    continue
            except Exception as exc:
                last_exc = exc
                break

        if resp is None:
            if last_exc is not None:
                logger.error("signin request failed: %s", last_exc)
            yield event.plain_result("签到失败：接口无法连接，请稍后再试")
            return

        if resp.status_code != 200:
            self._debug("http status=%s", resp.status_code)
            yield event.plain_result(f"签到失败：HTTP {resp.status_code}")
            return

        try:
            data = resp.json()
        except Exception:
            self._debug("json parse failed")
            yield event.plain_result("签到失败：接口返回异常")
            return

        message = data.get("message") or "签到失败：接口未返回消息"
        yield event.plain_result(str(message))

    def _format_players(self, players: list[Any], max_count: int) -> str:
        if not players:
            return "暂无玩家"

        players = sorted(players, key=lambda p: getattr(p, "score", 0), reverse=True)
        lines = []
        for p in players[:max_count]:
            name = (p.name or "玩家").strip() if hasattr(p, "name") else "玩家"
            score = getattr(p, "score", 0)
            duration = getattr(p, "duration", 0)
            total_seconds = int(duration)
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            lines.append(f"{name} | {score} | {minutes}:{seconds:02d}")
        return "\n".join(lines)

    def _format_seconds_short(self, value: Any) -> str:
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return "暂无"
            if ":" in text:
                parts = [p for p in text.split(":") if p != ""]
                if len(parts) in (2, 3) and all(p.isdigit() for p in parts):
                    nums = [int(p) for p in parts]
                    if len(nums) == 3:
                        total = nums[0] * 3600 + nums[1] * 60 + nums[2]
                    else:
                        total = nums[0] * 60 + nums[1]
                else:
                    total = None
            else:
                total = None
            if total is None:
                try:
                    total = int(text)
                except Exception:
                    return "暂无"
        else:
            try:
                total = int(value)
            except Exception:
                return "暂无"
        if total <= 0:
            return "0m"
        hours = total // 3600
        minutes = (total % 3600) // 60
        if hours > 0:
            return f"{hours}h{minutes:02d}m"
        return f"{minutes}m"

    def _is_bot_player(self, player: Any) -> bool:
        if getattr(player, "is_bot", False):
            return True
        name = (getattr(player, "name", "") or "").strip()
        if not name:
            return False
        low = name.lower()
        if low in {"bot", "[bot]", "(bot)", "computer", "npc"}:
            return True
        if low.startswith("bot") and len(low) <= 12:
            return True
        if low.endswith("bot") and len(low) <= 12:
            return True
        return False

    def _split_players(self, players: list[Any]) -> tuple[list[Any], list[Any]]:
        humans: list[Any] = []
        bots: list[Any] = []
        for p in players:
            if self._is_bot_player(p):
                bots.append(p)
            else:
                humans.append(p)
        return humans, bots

    def _format_players_html(self, players: list[Any], max_count: int) -> str:
        if not players:
            return '<div class="players-empty">暂无玩家</div>'

        players = sorted(players, key=lambda p: getattr(p, "score", 0), reverse=True)
        rows = []
        for idx, p in enumerate(players[:max_count], 1):
            name = (p.name or "玩家").strip() if hasattr(p, "name") else "玩家"
            score = getattr(p, "score", 0)
            duration = getattr(p, "duration", 0)
            minutes = int(duration) // 60
            rows.append(
                "<tr>"
                f"<td>{idx}</td>"
                f"<td>{html.escape(name)}</td>"
                f"<td>{score}</td>"
                f"<td>{minutes}m</td>"
                "</tr>"
            )
        rows_html = "".join(rows)
        return (
            "<table class=\"players-table\">"
            "<thead><tr><th>#</th><th>玩家</th><th>分数</th><th>时长</th></tr></thead>"
            f"<tbody>{rows_html}</tbody>"
            "</table>"
        )

    def _format_servers_list(self) -> str:
        servers = self._server_map()
        if not servers:
            return "未配置服务器列表"
        lines = ["当前服务器列表："]
        for name, addr in servers.items():
            lines.append(f"{name}: {addr}")
        return "\n".join(lines)

    async def _handle_stats_query(self, event: AstrMessageEvent):
        user_id = self._extract_user_id(event)
        if not user_id:
            yield event.plain_result("未获取到用户QQ号，无法查询")
            return

        now_ts = datetime.now().timestamp()
        cooldown_sec = 8.0
        expire_at = self._stats_cooldowns.get(user_id, 0.0)
        if expire_at > now_ts:
            yield event.plain_result("正在查询中，请勿重复查询！")
            return
        self._stats_cooldowns[user_id] = now_ts + cooldown_sec
        yield event.plain_result("正在查询数据，请稍后...")

        url = self._build_stats_url(user_id)
        timeout_ms = int(self._cfg("timeout_ms", 5000))
        timeout = max(timeout_ms, 1000) / 1000.0
        self._debug("stats request url=%s", url)

        resp = None
        last_exc: Exception | None = None
        client = self._get_http_client()
        for attempt in range(3):
            try:
                resp = await client.get(url, timeout=timeout)
                break
            except (httpx.TimeoutException, httpx.RequestError) as exc:
                last_exc = exc
                if attempt < 2:
                    await asyncio.sleep(0.6 * (attempt + 1))
                    continue
            except Exception as exc:
                last_exc = exc
                break

        if resp is None:
            if last_exc is not None:
                logger.error("stats request failed: %s", last_exc)
            yield event.plain_result("查询超时或网络异常，请稍后重试")
            return

        if resp.status_code != 200:
            yield event.plain_result(f"查询失败：HTTP {resp.status_code}")
            return

        try:
            payload = resp.json()
        except Exception:
            yield event.plain_result("查询失败：接口返回异常")
            return

        if not payload or not payload.get("success", False):
            message = payload.get("message") if isinstance(payload, dict) else None
            yield event.plain_result(str(message or "请先在网站绑定QQ与Steam账号"))
            return

        data = payload.get("data") or {}
        steam_id = str(data.get("steamId64") or "").strip()
        if not steam_id:
            yield event.plain_result("请先在网站绑定QQ与Steam账号")
            return

        credits = data.get("credits") or 0
        play_time = data.get("playTime") or {}
        player_name = str(play_time.get("playerName") or "").strip() or "玩家数据查询"
        stats = data.get("stats") or {}

        def _num(key: str) -> int:
            try:
                return int(stats.get(key) or 0)
            except Exception:
                return 0

        kills = _num("kills")
        first_blood = _num("firstBlood")
        deaths = _num("deaths")
        grenades = _num("grenades")
        shoots = _num("shoots")
        mvp = _num("mvp")
        round_win = _num("roundWin")
        round_lose = _num("roundLose")
        rounds_overall = _num("roundsOverall")
        rounds_total = rounds_overall or (round_win + round_lose)
        kd = None
        if kills > 0 or deaths > 0:
            kd_val = kills / max(1, deaths)
            kd = f"{kd_val:.2f}"
        winrate = None
        if round_win + round_lose > 0:
            winrate_val = round_win / (round_win + round_lose) * 100.0
            winrate = f"{winrate_val:.1f}%"

        def _pick_time(*keys: str) -> Any:
            for key in keys:
                if key in play_time and play_time.get(key) is not None:
                    return play_time.get(key)
            lowered = {str(k).lower(): v for k, v in play_time.items()}
            for key in keys:
                if key.lower() in lowered and lowered[key.lower()] is not None:
                    return lowered[key.lower()]
            return None

        total_time = self._format_seconds_short(_pick_time("totalTime", "all", "total"))
        ct_time = self._format_seconds_short(_pick_time("ctTime", "ct"))
        t_time = self._format_seconds_short(_pick_time("tTime", "TTime", "t", "t_time", "time_t", "terroristTime"))
        spec_time = self._format_seconds_short(_pick_time("specTime", "spec"))
        alive_time = self._format_seconds_short(_pick_time("aliveTime", "alive"))
        dead_time = self._format_seconds_short(_pick_time("deadTime", "dead"))
        if self._debug_enabled() and t_time == "暂无":
            try:
                self._debug("play_time keys=%s", list(play_time.keys()))
            except Exception:
                self._debug("play_time raw=%s", play_time)

        now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        response = (
            "玩家数据查询\n"
            "--------------------\n"
            f"玩家名称：{player_name}\n"
            f"QQ号：{user_id}\n"
            f"Steam64：{steam_id}\n"
            f"正式服积分：{credits}\n"
            f"K/D：{kd or '暂无'}\n"
            f"胜率：{winrate or '暂无'}\n"
            f"击杀数：{kills}\n"
            f"首杀数：{first_blood}\n"
            f"死亡数：{deaths}\n"
            f"投掷物次数：{grenades}\n"
            f"开火数：{shoots}\n"
            f"MVP次数：{mvp}\n"
            f"回合数：{rounds_total}\n"
            f"存活时长：{alive_time}\n"
            f"CT时长：{ct_time}\n"
            f"T时长：{t_time}\n"
            f"总时长：{total_time}\n"
            f"旁观时长：{spec_time}\n"
            f"死亡时长：{dead_time}\n"
            "--------------------\n"
            f"查询时间：{now_text}\n"
        )

        render_mode = str(self._cfg("stats_render_mode", "card")).strip().lower()
        if render_mode == "card":
            if not self._list_header_cache() and self._get_header_urls():
                yield event.plain_result("头图缓存为空，正在准备数据，请稍候...")
                await self._prefetch_headers(1, int(self._cfg("header_cache_limit", 30)))

            image_bytes = await self._render_stats_card_pil(
                title=player_name,
                qq_id=user_id,
                steam_id=steam_id,
                credits=credits,
                kd=kd,
                winrate=winrate,
                kills=kills,
                first_blood=first_blood,
                deaths=deaths,
                grenades=grenades,
                shoots=shoots,
                mvp=mvp,
                rounds_total=rounds_total,
                round_win=round_win,
                round_lose=round_lose,
                total_time=total_time,
                ct_time=ct_time,
                t_time=t_time,
                spec_time=spec_time,
                alive_time=alive_time,
                dead_time=dead_time,
                now_text=now_text,
            )

            if image_bytes and isinstance(event, AiocqhttpMessageEvent):
                base64_str = base64.b64encode(image_bytes).decode("utf-8")
                image_file_str = f"base64://{base64_str}"
                payload = {"message": [{"type": "image", "data": {"file": image_file_str}}]}
                try:
                    if event.is_private_chat():
                        payload["user_id"] = int(event.get_sender_id())
                        action = "send_private_msg"
                    else:
                        payload["group_id"] = int(event.get_group_id())
                        action = "send_group_msg"
                    await event.bot.api.call_action(action, **payload)
                    event.stop_event()
                    asyncio.create_task(self._silent_refresh_header_cache())
                    return
                except Exception as exc:
                    logger.error("stats card base64 send failed: %s", exc)

            self._debug("stats card render failed: no image bytes")

        yield event.plain_result(response)

    # --- stats card rendering ---
    async def _render_stats_card_pil(
        self,
        title: str,
        qq_id: str,
        steam_id: str,
        credits: Any,
        kd: str | None,
        winrate: str | None,
        kills: int,
        first_blood: int,
        deaths: int,
        grenades: int,
        shoots: int,
        mvp: int,
        rounds_total: int,
        round_win: int,
        round_lose: int,
        total_time: str,
        ct_time: str,
        t_time: str,
        spec_time: str,
        alive_time: str,
        dead_time: str,
        now_text: str,
    ) -> bytes | None:
        card_width, card_height = self._estimate_stats_card_size(
            title,
            qq_id,
            steam_id,
            credits,
            kd,
            winrate,
            kills,
            first_blood,
            deaths,
            grenades,
            shoots,
            mvp,
            rounds_total,
            round_win,
            round_lose,
            total_time,
            ct_time,
            t_time,
            spec_time,
            alive_time,
            dead_time,
        )
        header_image = await self._get_header_image(card_width, card_height)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            partial(
                self._render_stats_card_pil_sync,
                title,
                qq_id,
                steam_id,
                credits,
                kd,
                winrate,
                kills,
                first_blood,
                deaths,
                grenades,
                shoots,
                mvp,
                rounds_total,
                round_win,
                round_lose,
                total_time,
                ct_time,
                t_time,
                spec_time,
                alive_time,
                dead_time,
                now_text,
                header_image,
            ),
        )

    def _render_stats_card_pil_sync(
        self,
        title: str,
        qq_id: str,
        steam_id: str,
        credits: Any,
        kd: str | None,
        winrate: str | None,
        kills: int,
        first_blood: int,
        deaths: int,
        grenades: int,
        shoots: int,
        mvp: int,
        rounds_total: int,
        round_win: int,
        round_lose: int,
        total_time: str,
        ct_time: str,
        t_time: str,
        spec_time: str,
        alive_time: str,
        dead_time: str,
        now_text: str,
        header_image: Image.Image | None,
    ) -> bytes | None:
        card_width = 820
        padding = 28
        outer_margin = 0
        corner_radius = 0
        bg_color = (36, 36, 36)
        overlay_color = (0, 0, 0, 160)
        line_color = (85, 85, 85)
        text_color = (245, 245, 245)
        label_color = (178, 184, 190)
        muted_color = (160, 166, 173)

        title_font = self._load_font(26, bold=True)
        section_font = self._load_font(20, bold=True)
        label_font = self._load_font(18, bold=True)
        value_font = self._load_font(18, bold=False)
        small_font = self._load_font(14, bold=False)

        title_text = title or "玩家数据查询"

        account_items = [
            ("玩家名称", title_text),
            ("QQ号", qq_id),
            ("Steam64", steam_id),
            ("正式服积分", str(credits)),
        ]
        stats_items = [
            ("K/D", kd or "暂无"),
            ("胜率", winrate or "暂无"),
            ("击杀数", str(kills)),
            ("首杀数", str(first_blood)),
            ("死亡数", str(deaths)),
            ("投掷物", str(grenades)),
            ("开火数", str(shoots)),
            ("MVP", str(mvp)),
            ("回合数", str(rounds_total)),
            ("胜/负", f"{round_win}/{round_lose}"),
        ]
        time_items = [
            ("存活时长", alive_time),
            ("CT时长", ct_time),
            ("T时长", t_time),
            ("总时长", total_time),
            ("旁观时长", spec_time),
            ("死亡时长", dead_time),
        ]

        col_gap = 36
        col_width = (card_width - padding * 2 - col_gap) // 2
        label_min_w = 120
        value_gap = 16
        line_height = 26
        title_height = title_font.getbbox("测")[3]
        section_title_height = section_font.getbbox("测")[3]
        footer_height = small_font.getbbox("测")[3] + 8
        divider_gap = 12
        time_gap = divider_gap * 3
        bottom_padding = 12

        def _section_height(items: list[tuple[str, str]]) -> int:
            rows = (len(items) + 1) // 2
            return section_title_height + 6 + rows * line_height

        content_height = (
            title_height
            + 8
            + divider_gap
            + _section_height(account_items)
            + divider_gap * 2 + 2
            + _section_height(stats_items)
            + divider_gap * 2 + 2
            + _section_height(time_items)
            + divider_gap
            + time_gap
            + footer_height
            + bottom_padding
        )
        card_height = padding * 2 + content_height

        base = Image.new(
            "RGBA",
            (card_width + outer_margin * 2, card_height + outer_margin * 2),
            bg_color,
        )
        card = Image.new("RGBA", (card_width, card_height), (0, 0, 0, 0))

        bg = header_image
        if bg is not None and (bg.width != card_width or bg.height != card_height):
            bg = self._crop_cover(bg, card_width, card_height)
        if bg is None:
            bg = Image.new("RGB", (card_width, card_height), (58, 58, 58))
        bg_rgba = bg.convert("RGBA")
        bg_blur = bg_rgba.filter(ImageFilter.GaussianBlur(2.4))
        bg_rgba = Image.blend(bg_rgba, bg_blur, 0.42)
        overlay = Image.new("RGBA", (card_width, card_height), overlay_color)
        bg_rgba = Image.alpha_composite(bg_rgba, overlay)

        mask = Image.new("L", (card_width, card_height), 0)
        ImageDraw.Draw(mask).rounded_rectangle(
            [0, 0, card_width, card_height],
            radius=corner_radius,
            fill=255,
        )
        card.paste(bg_rgba, (0, 0), mask)

        # 取消面板毛玻璃，仅保留整体轻微模糊

        draw = ImageDraw.Draw(card)
        y = padding
        draw.text((padding, y), title_text, fill=text_color, font=title_font)
        y += title_height + 8
        draw.line((padding, y, card_width - padding, y), fill=line_color, width=2)
        y += divider_gap
        account_layout = self._section_layout(account_items, label_font, value_font, col_width, label_min_w, value_gap)
        stats_layout = self._section_layout(stats_items, label_font, value_font, col_width, label_min_w, value_gap)
        time_layout = self._section_layout(time_items, label_font, value_font, col_width, label_min_w, value_gap)

        y = self._draw_two_col_section(
            draw,
            y,
            "账号信息",
            account_items,
            col_width,
            label_font,
            value_font,
            section_font,
            label_color,
            text_color,
            padding,
            col_gap,
            line_height,
            value_gap,
            account_layout,
        )
        y += divider_gap
        draw.line((padding, y, card_width - padding, y), fill=line_color, width=1)
        y += divider_gap
        y = self._draw_two_col_section(
            draw,
            y,
            "数据统计",
            stats_items,
            col_width,
            label_font,
            value_font,
            section_font,
            label_color,
            text_color,
            padding,
            col_gap,
            line_height,
            value_gap,
            stats_layout,
        )
        y += divider_gap
        draw.line((padding, y, card_width - padding, y), fill=line_color, width=1)
        y += divider_gap
        y = self._draw_two_col_section(
            draw,
            y,
            "时长统计",
            time_items,
            col_width,
            label_font,
            value_font,
            section_font,
            label_color,
            text_color,
            padding,
            col_gap,
            line_height,
            value_gap,
            time_layout,
        )
        footer_font = self._load_font(12, bold=False)
        footer_h = footer_font.getbbox("测")[3] + 2
        footer_y = card_height - bottom_padding - footer_h
        line_y = footer_y - max(2, int(divider_gap * 0.6))
        draw.line((padding, line_y, card_width - padding, line_y), fill=line_color, width=1)
        draw.text((padding, footer_y), f"查询时间：{now_text}", fill=muted_color, font=footer_font)

        base.paste(card, (outer_margin, outer_margin), card)
        out = BytesIO()
        base.convert("RGB").save(out, format="JPEG", quality=90)
        return out.getvalue()


    # --- request handlers ---
    async def _handle_a2s_query(self, event: AstrMessageEvent, name: str, addr: str):
        try:
            host, port = self._parse_address(addr)
        except ValueError as exc:
            yield event.plain_result(f"服务器配置错误：{exc}")
            return

        timeout_ms = int(self._cfg("a2s_timeout_ms", 3000))
        timeout = max(timeout_ms, 1000) / 1000.0
        api_timeout = self._server_status_api_timeout(timeout_ms)
        api_base = self._server_status_api_base_url()
        api_fallback = self._server_status_api_fallback()

        self._debug("a2s query name=%s addr=%s:%s", name, host, port)

        info = None
        players: list[Any] = []
        if api_base:
            result = await self._fetch_server_status_api(host, port, api_timeout)
            if result:
                info, players = result
            elif not api_fallback:
                yield event.plain_result("服务器查询失败，请稍后再试")
                return

        if info is None:
            try:
                info = await a2s.ainfo((host, port), timeout=timeout)
                players = await a2s.aplayers((host, port), timeout=timeout)
            except Exception as exc:
                logger.error("a2s query failed: %s", exc)
                yield event.plain_result("服务器查询失败，请稍后再试")
                return

        render_mode = str(self._cfg("a2s_render_mode", "image_text")).strip().lower()
        max_players_show = int(self._cfg("players_limit", 10))
        human_players, bot_players = self._split_players(players)
        bot_count = len(bot_players)
        players_md = self._format_players(human_players, max_players_show)

        now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        response = (
            "服务器状态更新！\n"
            "--------------------\n"
            f"服务器名称：{info.server_name}\n"
            "官方网站：example.com\n"
            f"服务器IP：{host}:{port}\n"
            f"服务器人数：{info.player_count}/{info.max_players}"
            f"{f' ({bot_count} BOT)' if bot_count else ''}\n"
            f"当前地图：{info.map_name}\n"
            f"当前延迟：{info.ping * 1000:.0f} ms\n"
            "--------------------\n"
            "分数 | 玩家 | 时间\n"
            f"{players_md}\n"
            "--------------------\n"
            f"查询时间：{now_text}\n"
        )

        if render_mode == "card":
            if not self._list_header_cache() and self._get_header_urls():
                yield event.plain_result("头图缓存为空，正在准备数据，请稍候...")
                await self._prefetch_headers(1, int(self._cfg("header_cache_limit", 30)))
            image_bytes = await self._render_card_pil(
                host=host,
                port=port,
                server_alias=name,
                info=info,
                players=players,
                now_text=now_text,
                max_players_show=max_players_show,
            )
            if image_bytes and isinstance(event, AiocqhttpMessageEvent):
                base64_str = base64.b64encode(image_bytes).decode("utf-8")
                image_file_str = f"base64://{base64_str}"
                payload = {"message": [{"type": "image", "data": {"file": image_file_str}}]}
                try:
                    if event.is_private_chat():
                        payload["user_id"] = int(event.get_sender_id())
                        action = "send_private_msg"
                    else:
                        payload["group_id"] = int(event.get_group_id())
                        action = "send_group_msg"
                    await event.bot.api.call_action(action, **payload)
                    event.stop_event()
                    asyncio.create_task(self._silent_refresh_header_cache())
                    return
                except Exception as exc:
                    logger.error("card base64 send failed: %s", exc)

            self._debug("card render failed: no image bytes")

        if render_mode == "image_text":
            header_url = str(self._cfg("header_image_url", "")).strip()
            if header_url and self._is_http_url(header_url):
                chain = [
                    Comp.Image.fromURL(header_url),
                    Comp.Plain(response),
                ]
                yield event.chain_result(chain)
                return
            yield event.plain_result(response)
            return

        yield event.plain_result(response)

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def on_group_message(self, event: AstrMessageEvent):
        text = (event.message_str or "").strip()
        text_lower = text.lower()

        rcon_prefix = self._rcon_prefix()
        rcon_request: tuple[str, str] | None = None
        if rcon_prefix:
            prefix_lower = rcon_prefix.lower()
            if text_lower.startswith(prefix_lower + " "):
                parts = text.split(maxsplit=2)
                if len(parts) < 3:
                    yield event.plain_result(f"用法：{rcon_prefix} <别名> <命令>")
                    return
                rcon_request = (parts[1], parts[2])

        match = self._match_server_alias(text)
        is_signin = text == "签到"
        is_ip_list = text_lower == "ip"
        stats_keywords = self._stats_keywords()
        stats_keyword_set = {k.lower() for k in stats_keywords}
        is_stats = text in stats_keywords or text_lower in stats_keyword_set
        if not is_signin and not match and not is_ip_list and not is_stats and not rcon_request:
            return

        self._debug("recv group message text=%r", text)

        if not self._is_strict_single_plain(event):
            self._debug("skip: not strict single plain")
            return

        group_id = self._extract_group_id(event)
        self._debug("group_id=%s allowed=%s", group_id, list(self._allowed_groups()))
        if not self._is_group_allowed(group_id):
            self._debug("skip: group not allowed")
            return

        if rcon_request:
            alias, command = rcon_request
            match_alias = self._match_server_alias(alias)
            if not match_alias:
                yield event.plain_result("未找到服务器别名")
                return
            name, addr = match_alias
            async for result in self._handle_rcon_command(event, name, addr, command):
                yield result
            return

        if is_ip_list:
            yield event.plain_result(self._format_servers_list())
            return

        if is_signin:
            async for result in self._handle_signin(event):
                yield result
            return

        if is_stats:
            async for result in self._handle_stats_query(event):
                yield result
            return

        if match:
            name, addr = match
            async for result in self._handle_a2s_query(event, name, addr):
                yield result

    async def terminate(self):
        if self._http is not None:
            await self._http.aclose()
            self._http = None


