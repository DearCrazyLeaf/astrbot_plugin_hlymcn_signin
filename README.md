# HLYM Server Toolkit (AstrBot Plugin)

### 群聊签到 + 服务器查询（A2S/后端API）+ 数据查询 + RCON 指令

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![AstrBot](https://img.shields.io/badge/AstrBot-v4.12%2B-brightgreen)
![License](https://img.shields.io/badge/License-GPL--3.0-orange)

![views](https://count.getloli.com/get/@astrbotchuanhuatong?theme=booru-jaypee)

---

## ✨ 简介 | Introduction

这是一个为 **AstrBot** 编写的多功能服务器工具插件，支持：
- 群聊签到（HTTP API）
- 服务器查询（后端 API 或直连 A2S）
- 玩家列表渲染（卡片 / 文字）
- 数据统计查询（卡片 / 文字）
- RCON 远程指令（后端 API）

插件面向 QQ 群聊场景设计，支持 WebUI 配置，无需 @ 机器人

> [!IMPORTANT]  
> 签到 / 玩家数据查询 / 统计等功能依赖你自行编写并适配的后端服务，本仓库不提供后端实现

---

## ✅ 功能列表 | Features

- **签到**：发送 `签到` 即可触发后端签到接口
- **查服**：发送服务器别名触发查询（例如 `cs`）
- **查数据**：发送 `data` / `stat` 等关键词查询个人数据
- **RCON**：在群聊中直接远程执行命令
- **卡片渲染**：支持图片卡片输出（A2S / 统计）
- **可配置**：全部参数在 WebUI 配置，无需改代码

---

## 📦 安装 | Installation

将插件目录放入：
```
AstrBot/data/plugins/astrbot_plugin_hlymcn_signin
```

重启 AstrBot 后即可在 WebUI 中看到插件

---

## ⚙️ 配置 | Configuration

插件使用 `_conf_schema.json` 定义配置，配置入口：
```
AstrBot WebUI -> 插件 -> 插件配置
```

### 🔧 核心配置

| 配置项 | 说明 |
|--------|------|
| base_url | 后端基础地址（签到 / 数据查询） |
| signin_path | 签到接口路径 |
| stats_path | 数据查询接口路径 |
| server_status_api_base_url | 查服 API 基础地址（留空默认用 base_url） |
| server_status_api_path | 查服 API 路径（可自定义，默认 <YOUR_SERVER_STATUS_API_PATH>） |
| server_status_api_fallback | 查服 API 失败是否回退 A2S 直连 |
| servers | 服务器别名列表（别名=ip:port） |
| a2s_render_mode | 查服输出模式（card / image_text / text） |
| stats_render_mode | 数据输出模式（card / text） |

### 🖼️ 头图配置

| 配置项 | 说明 |
|--------|------|
| header_image_url | 头图链接（支持多条，用换行/逗号/分号分隔） |
| header_prefetch_count | 预载图片数量 |
| header_cache_limit | 头图缓存上限 |
| header_image_max_bytes | 头图下载最大字节数（默认 3MB） |

### ✅ RCON 配置

| 配置项 | 说明 |
|--------|------|
| rcon_api_base_url | RCON API 基础地址 |
| rcon_password | RCON 密码 |
| rcon_command_prefix | RCON 指令前缀（默认 rcon） |
| rcon_admin_ids | 允许使用 RCON 的管理员 QQ 列表（为空则禁用 RCON） |

---

## 💬 使用方法 | Usage

### 签到
```
签到
```

### 查服
```
cs
```
（其中 `cs` 为配置里的服务器别名）

### 查数据
```
data
stat
我的数据
查询数据
```

### RCON
```
rcon <别名> <命令>
```
示例：
```
rcon cs status
```

---

## 🧩 输出模式 | Render Modes

- `card`：图片卡片（推荐）
- `image_text`：头图 + 文字
- `text`：纯文本输出

---

## 🛡️ 注意事项 | Notes

- 查服 API 需要支持 `<YOUR_SERVER_STATUS_API_PATH>` 接口
- 若 API 不稳定可开启 `server_status_api_fallback` 使用 A2S 直连
- RCON 功能需后端支持对应 API

---

## 📄 License

GPL-3.0

---

> Made for AstrBot ❤️
