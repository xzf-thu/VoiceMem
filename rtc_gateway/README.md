# VoiceMem RTC Gateway

一对一本机 RTC 媒体网关。

- 浏览器与网关：WebRTC / Opus / SRTP
- 网关与 Python：本机 WebSocket IPC（Opus）
- 字幕、记忆和业务事件：原 FastAPI WebSocket
- 默认播放缓冲：200ms；`answer_interrupt` 会清空 Python 和网关两级队列

构建：

```bash
cd rtc_gateway
go build -o voicemem-rtc .
```

需要 Go 1.24 或更高版本。

`web/run.py --transport rtc` 会自动启动构建好的网关。HTTP IPC 只监听
`127.0.0.1:8790`；ICE UDP 和 ICE TCP 监听本机网卡的 `8791` 和 `8792`
端口，只发布设备的本地网络候选，不配置公网地址映射。

## Local media gateway

The gateway carries one WebRTC audio session between a browser and the VoiceMem
process. Browser media uses Opus over SRTP. The Python IPC carries Opus over a
local WebSocket, while application events stay on the FastAPI WebSocket.

Build with Go 1.24 or newer:

```bash
go build -o voicemem-rtc .
```

`web/run.py --transport rtc` starts the binary automatically. HTTP IPC binds to
`127.0.0.1:8790`. ICE UDP and ICE TCP use ports `8791` and `8792` on local
network interfaces and publish local network candidates only.
