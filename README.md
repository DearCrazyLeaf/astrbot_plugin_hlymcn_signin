# HLYM Server Toolkit (AstrBot Plugin)

### HLYM 社区服务器工具插件（签到 / 查服 / 数据查询 / RCON / MC 查询）
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![AstrBot](https://img.shields.io/badge/AstrBot-v4.12%2B-brightgreen)
![License](https://img.shields.io/badge/License-GPL--3.0-orange)

![views](https://count.getloli.com/get/@astrbotchuanhuatong?theme=booru-jaypee)

---

## ✨ 简介 | Introduction

这是一个为 **AstrBot** 编写的服务器工具插件，聚焦群内高频运维场景：

- ✅ 群聊签到（依赖后端）
- ✅ 玩家数据查询（依赖后端）
- ✅ CS2/A2S 查服（可本地直查，也可后端增强）
- ✅ Minecraft 查服（独立配置，不走全局后端）
- ✅ 本地 RCON 直连执行（不依赖后端 RCON 转发）
- ✅ 图文卡片输出、头图缓存、地图封面回退

> [!IMPORTANT]
> 本插件是“前端能力层”，**高度依赖后端接口约定**。  
> 仓库仅提供 AstrBot 插件实现，不提供通用后端服务。  
> 若你没有对接自己的后端，建议先使用“本地可用功能”（A2S/MC 查询、本地 RCON），再逐步接入签到与数据查询接口。

[![Release](https://img.shields.io/github/v/release/DearCrazyLeaf/astrbot_plugin_hlymcn_signin?include_prereleases&color=blueviolet&label=最新版本)](https://github.com/DearCrazyLeaf/astrbot_plugin_hlymcn_signin/releases/latest)
[![License](https://img.shields.io/badge/许可证-GPL%203.0-orange)](https://www.gnu.org/licenses/gpl-3.0.txt)
[![Issues](https://img.shields.io/github/issues/DearCrazyLeaf/astrbot_plugin_hlymcn_signin?color=darkgreen&label=反馈)](https://github.com/DearCrazyLeaf/astrbot_plugin_hlymcn_signin/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/DearCrazyLeaf/astrbot_plugin_hlymcn_signin?color=blue&label=请求)](https://github.com/DearCrazyLeaf/astrbot_plugin_hlymcn_signin/pulls)
[![GitHub Stars](https://img.shields.io/github/stars/DearCrazyLeaf/astrbot_plugin_hlymcn_signin?color=yellow&label=标星)](https://github.com/DearCrazyLeaf/astrbot_plugin_hlymcn_signin/stargazers)

---

## 🧩 功能矩阵 | Capability Matrix

| 功能 | 是否依赖后端 | 说明 |
|------|--------------|------|
| A2S / CS2 查服（直连） | 否 | 使用 `a2s` 直接查询 `ip:port` |
| A2S / CS2 查服（后端增强） | 是（可选） | 使用批量查服 API（延迟、玩家、地图封面等） |
| MC 查服 | 否 | 使用独立 MC API 配置，不走全局后端 |
| 本地 RCON | 否 | 插件内直接 TCP RCON，管理员白名单控制 |
| 群聊签到 | 是 | 调用你自己的签到后端 |
| 玩家数据查询 | 是 | 调用你自己的数据统计后端 |

---

## 📦 安装 | Installation

将插件目录放入：

```text
AstrBot/data/plugins/astrbot_plugin_hlymcn_signin
```

重启 AstrBot 后在 WebUI 中启用插件。

---

## ⚙️ 配置结构 | Configuration

插件已按模块分组，WebUI 中更易定位：

```text
AstrBot WebUI -> 插件 -> HLYM服务器工具 -> 插件配置
```

### 1) `basic_settings`（基础设置）

- `allowed_group_ids`：允许触发的群号列表
- `strict_single_plain`：仅纯文本消息触发（建议开启）
- `stats_keywords`：数据查询关键词（可多关键词）

### 2) `local_server_tools`（本地查服与展示）

- `servers`：查服别名映射，格式 `别名=ip:port`
- `players_limit`：卡片/文本显示玩家数量
- `a2s_render_mode`：`image_text` / `card` / `text`
- `a2s_timeout_ms`：A2S 查询超时
- `server_official_site`：默认官网显示值
- `server_official_sites`：按别名覆盖官网，格式 `别名=网站`
- `server_query_bg_mode`：
  - `header`：使用自定义头图
  - `map_cover`：优先使用后端地图封面，失败自动回退头图

### 3) `minecraft_query`（MC 独立查询）

- `mc_servers`：MC 别名映射，格式 `别名=host:port`（缺省端口默认 `25565`）
- `mc_query_api_base`：MC 查询 API 根地址
- `mc_timeout_ms`：查询超时
- `mc_render_mode`：`text` / `image_text`
- `mc_show_address`：是否在结果中显示服务器地址

MC 输出已做适配：

- 自动清理 MOTD 中 `§` 颜色控制符
- 玩家列表按纵向列表展示（便于阅读）
- 地址显示可通过开关控制

### 4) `local_rcon`（本地 RCON）

- `rcon_admin_ids`：管理员 QQ 列表（为空则禁用 RCON）
- `rcon_command_prefix`：命令前缀，默认 `rcon`
- `rcon_password`：全局密码
- `rcon_passwords`：服务器专属密码映射（`别名=密码` 或 `ip:port=密码`）
- `rcon_timeout_ms`：RCON 超时

### 5) `header_settings`（头图与缓存）

- `header_image_url`：头图 URL（支持多条）
- `header_prefetch_count`：预载数量
- `header_cache_limit`：缓存上限
- `header_image_max_bytes`：下载大小限制

### 6) `backend_signin_stats`（后端：签到与数据查询）

- `base_url`：后端基础地址
- `signin_path`：签到接口路径
- `stats_path`：数据查询接口路径
- `timeout_ms`：请求超时
- `device_type` / `signin_type`：透传参数
- `stats_render_mode`：`card` / `text`

### 7) `backend_server_query`（后端：服务器状态增强）

- `server_status_api_base_url`：查服后端地址（留空则复用 `base_url`）
- `server_status_api_path`：查服接口路径
- `server_status_api_timeout_ms`：查服接口超时
- `server_status_api_fallback`：后端失败时是否回退 A2S 直连

### 8) `debug_settings`

- `debug_log`：控制台输出调试日志

---

## 🚀 使用方式 | Usage

### 常用指令

| 指令 | 说明 |
|------|------|
| `签到` | 触发签到流程（需后端） |
| `ip` | 列出已配置服务器别名（A2S + MC） |
| `<A2S别名>` | 查询该 A2S/CS2 服务器 |
| `<MC别名>` | 查询该 MC 服务器 |
| `rcon <别名> <命令>` | 直连执行 RCON（需管理员） |
| `stats_keywords` 中任一词 | 玩家数据查询（需后端） |
| `预载头图` / `prefetch_header` | 手动预载头图缓存 |

---

## 🖼️ 背景图策略说明 | Background Strategy

当 `server_query_bg_mode=map_cover` 时：

1. 插件优先从后端返回数据中读取地图封面 URL  
2. 若后端无封面或下载失败，自动回退到自定义头图  
3. 回退后仍可正常出图，不影响主流程

> [!TIP]
> 若你使用 OSS 作为地图封面缓存，建议由后端保证“先查 OSS、再尝试拉取并写回 OSS”的策略，插件侧只消费最终 URL。

---

## 🔌 后端编写说明 | Backend Contract

如果你要启用“签到 / 数据查询 / 后端增强查服”，后端至少需要满足以下接口约定。

### 1) 签到接口（`backend_signin_stats`）

- 请求：`GET {base_url}{signin_path}`
- Query 参数：
  - `userId`：QQ 号（插件自动传）
  - `deviceType`：来自配置 `device_type`
  - `signinType`：来自配置 `signin_type`
- 返回要求：
  - HTTP `200`
  - JSON 中至少包含 `message` 字段（插件会直接回显给用户）

示例：

```json
{
  "message": "签到成功，获得 10 积分"
}
```

### 2) 数据查询接口（`backend_signin_stats`）

- 请求：`GET {base_url}{stats_path}/{qq}`
- 返回要求：
  - HTTP `200`
  - 顶层建议结构：`success + message + data`
  - `success=false` 时，插件回显 `message`
  - `success=true` 时，`data` 中建议包含：
    - `steamId64`
    - `credits`
    - `playTime`（含 `playerName`、各时长字段）
    - `stats`（击杀、回合、胜负等统计字段）

示例：

```json
{
  "success": true,
  "message": "ok",
  "data": {
    "steamId64": "7656119xxxxxxxxxx",
    "credits": 1234,
    "playTime": {
      "playerName": "HLYM Player"
    },
    "stats": {
      "killCount": 100,
      "deathCount": 80,
      "roundCount": 50
    }
  }
}
```

### 3) 服务器状态增强接口（`backend_server_query`）

- 请求：`GET {server_status_api_base_url}{server_status_api_path}?servers=ip:port`
- 返回要求：
  - HTTP `200`
  - 顶层包含 `servers` 数组
  - 每个 server 项建议包含：
    - `name`、`map`
    - `raw.playerCount`、`raw.maxPlayers`、`raw.botCount`、`raw.game`
    - `query.ping`（毫秒）
    - `players[]` / `bots[]`（`name`、`score`、`time`）
    - 可选地图封面字段：`mapImage` / `map_image` / `basic_info.map_image`

示例（简化）：

```json
{
  "servers": [
    {
      "name": "HLYM Server",
      "map": "de_mirage",
      "raw": {
        "playerCount": 12,
        "maxPlayers": 32,
        "botCount": 0,
        "game": "Counter-Strike 2"
      },
      "query": {
        "ping": 45
      },
      "players": [
        { "name": "PlayerA", "score": 20, "time": 1200 }
      ],
      "mapImage": "https://your-oss/map/de_mirage.jpg"
    }
  ]
}
```

### 4) 地图封面模式说明

- 当 `server_query_bg_mode=map_cover`：
  - 插件优先使用后端返回的地图封面 URL；
  - 后端无封面/请求失败时，插件自动回退到自定义头图；
  - 若开启 `server_status_api_fallback=true`，后端失败后还会回退到 A2S 直连查询（文本数据仍可用）。

> [!NOTE]
> `minecraft_query` 是独立模块，不依赖上述后端接口，使用 `mc_query_api_base` 直查。

---

## 🧪 调试建议 | Debug Tips

开启 `debug_log` 后，可在控制台快速定位：

- 消息是否命中触发条件
- 群权限是否拦截
- 后端接口是否超时/失败
- 地图封面是否命中与回退
- RCON 管理员权限与连接错误

---

## 📄 License

GPL-3.0

---

> Made for AstrBot ❤️
