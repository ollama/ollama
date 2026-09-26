# TVBox Live Streaming Sources

This directory contains configuration files and a web-based player for IPTV/live streaming that can be used on Android TV.

## Files

- `live.m3u` - M3U playlist file with all streaming sources
- `config.json` - TVBox JSON configuration file for Android TV apps
- `player.html` - Web-based IPTV player that works on Android TV browsers
- `README.md` - This file

## 🎯 如何在安卓TV上使用 (How to Use on Android TV)

### 方法1: 使用 TVBox 應用 (Using TVBox App)

1. 在安卓TV上安裝 TVBox 應用
2. 打開 TVBox，進入配置設定
3. 添加配置地址或直接導入 `config.json` 文件
4. 配置包含所有直播源，包括：
   - 直播源倉庫 (范明明、YueChan、PTV研究所等)
   - 單線直播源
   - 文字源

### 方法2: 使用 TiviMate 或其他 IPTV 播放器 (Using TiviMate or Other IPTV Players)

1. 在安卓TV上安裝 TiviMate 或其他支援 M3U 的 IPTV 播放器
2. 在播放器中添加播放列表
3. 輸入 M3U 文件的 URL 或直接導入 `live.m3u` 文件
4. 開始觀看直播

### 方法3: 使用網頁播放器 (Using Web Player)

1. 將 `player.html` 文件上傳到網頁伺服器或通過 GitHub Pages 託管
2. 在安卓TV瀏覽器中打開網頁地址
3. 使用遙控器上下鍵選擇頻道
4. 按確認鍵開始播放

**功能特點:**
- 📺 支援 M3U 播放列表和直接流
- 🎮 支援電視遙控器導航
- ⛶ 全螢幕播放
- 🔄 重新加載功能
- 📱 響應式設計

## 🛠️ 配置說明 (Configuration Guide)

### config.json 格式

TVBox 配置文件包含以下結構:
- `lives`: 直播源列表
- `epg`: 電子節目指南 URL
- `logo`: 頻道標誌 URL
- `playerType`: 播放器類型 (0=系統播放器, 1=IJK播放器, 2=Exo播放器)

### 添加自定義源

您可以編輯 `config.json` 或 `live.m3u` 文件來添加自己的直播源:

**M3U 格式:**
```
#EXTINF:-1 group-title="分組名稱", 頻道名稱
http://example.com/stream.m3u8
```

**JSON 格式:**
```json
{
  "name": "頻道名稱",
  "type": 0,
  "url": "http://example.com/stream.m3u8",
  "playerType": 1,
  "epg": "",
  "logo": ""
}
```


## 📡 直播源列表 (Streaming Sources)

The configuration includes sources from:
- **范明明 (Fanmingming)** - IPv6, Global, and Radio streams
- **YueChan IPTV** - Comprehensive IPTV channels
- **1986 直播** - Live streaming sources
- **PTV研究所** - Comprehensive and Sports channels
- **AKTV** - Additional live channels
- **夢澤河** - TiviMate optimized sources
- **單線直播** - Direct streaming links (雲星家庭, 雲星嗶哩, 螞蟻論壇, 不良帥)
- **文字源** - Text-based sources (365直播, 吾愛評測)

## 🖥️ 技術規格 (Technical Specifications)

### 支援的格式 (Supported Formats)
- M3U/M3U8 播放列表
- HTTP/HTTPS 直播流
- HLS (HTTP Live Streaming)

### 支援的應用程式 (Compatible Applications)
- **TVBox** - 安卓TV專用應用
- **TiviMate** - 專業IPTV播放器
- **Kodi** - 媒體中心
- **VLC Media Player** - 通用播放器
- 任何支援M3U的IPTV播放器

### 網頁播放器功能 (Web Player Features)
- 遙控器友好的界面設計
- 自動播放支援
- 全螢幕模式
- 錯誤處理和重試機制
- M3U播放列表解析
- 頻道分組顯示

## Important Legal Notice

**WARNING**: Many of these streaming sources may be unauthorized IPTV sources that could violate copyright laws in your jurisdiction. Users are solely responsible for ensuring their use of these sources complies with all applicable laws and regulations. The availability, legality, and content of these sources may vary significantly by region.

Please respect copyright and licensing when using these streaming sources. Use at your own risk and discretion.
