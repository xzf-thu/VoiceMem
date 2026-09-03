"""Media transports for the demo: original WebSocket PCM and VoiceMem RTC."""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from contextlib import suppress
from fractions import Fraction


class WebSocketTransport:
    kind = "websocket"

    def __init__(self, socket):
        self.socket = socket

    async def start(self):
        pass

    def session_info(self):
        return {"transport": self.kind}

    async def receive(self):
        return await self.socket.receive()

    async def send_json(self, payload):
        await self.socket.send_json(payload)

    async def send_bytes(self, audio):
        await self.socket.send_bytes(audio)

    async def close(self):
        pass


class RTCTransport:
    """Pion owns WebRTC; this adapter owns PCM/Opus and keeps business on WS."""
    kind = "rtc"

    def __init__(self, socket, sample_rate=24000):
        self.socket, self.sample_rate = socket, sample_rate
        self.session_id = uuid.uuid4().hex
        self.gateway = os.environ.get("VOICEMEM_RTC_GATEWAY", "http://127.0.0.1:8790").rstrip("/")
        self._incoming = asyncio.Queue(maxsize=200)
        self._tasks = set()
        self._media = None
        self._decoder = self._encoder = self._resample_in = self._resample_out = None
        self._pcm = bytearray()
        self._frame_bytes = self.sample_rate // 50 * 2
        self._pcm_cond = asyncio.Condition()
        self._playing = False
        self._closed = False
        self._drain_task = None

    def _task(self, coro):
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def start(self):
        import av, websockets
        from av import AudioResampler
        self._decoder = av.CodecContext.create("opus", "r")
        self._encoder = av.CodecContext.create("libopus", "w")
        self._encoder.sample_rate = 48000
        self._encoder.layout = "mono"
        self._encoder.format = "s16"
        self._encoder.bit_rate = 64000
        self._encoder.time_base = Fraction(1, 48000)
        self._resample_in = AudioResampler(format="s16", layout="mono", rate=self.sample_rate)
        self._resample_out = AudioResampler(format="s16", layout="mono", rate=48000, frame_size=960)
        media_url = self.gateway.replace("http://", "ws://").replace("https://", "wss://")
        self._media = await websockets.connect(f"{media_url}/media/{self.session_id}", max_size=None)
        self._task(self._read_control())
        self._task(self._read_media())
        self._task(self._pace_output())

    def session_info(self):
        return {"transport": self.kind, "rtc": {"session_id": self.session_id}}

    async def _read_control(self):
        try:
            while not self._closed:
                await self._incoming.put(await self.socket.receive())
        except Exception:
            if not self._closed:
                await self._incoming.put({"type": "websocket.disconnect"})

    async def _read_media(self):
        import av
        try:
            async for data in self._media:
                if not isinstance(data, bytes) or not data or data[0] != 1:
                    continue
                for frame in self._decoder.decode(av.Packet(data[1:])):
                    for out in self._resample_in.resample(frame):
                        size = out.samples * 2
                        await self._incoming.put({"bytes": bytes(out.planes[0])[:size]})
        except Exception as e:
            if not self._closed:
                print(f"[rtc] media input stopped: {type(e).__name__}: {e}", flush=True)

    async def _pace_output(self):
        import av
        next_at = time.monotonic()
        while not self._closed:
            async with self._pcm_cond:
                while len(self._pcm) < self._frame_bytes and not self._closed:
                    await self._pcm_cond.wait()
                if self._closed:
                    return
                raw = bytes(self._pcm[:self._frame_bytes])
                del self._pcm[:self._frame_bytes]
                self._pcm_cond.notify_all()
            now = time.monotonic()
            next_at = now if next_at < now - .04 else next_at + .02
            await asyncio.sleep(max(0, next_at-now))
            frame = av.AudioFrame(format="s16", layout="mono", samples=self._frame_bytes // 2)
            frame.planes[0].update(raw)
            frame.sample_rate = self.sample_rate
            for f48 in self._resample_out.resample(frame):
                for packet in self._encoder.encode(f48):
                    await self._media.send(bytes([2]) + bytes(packet))

    async def receive(self):
        return await self._incoming.get()

    async def send_bytes(self, audio):
        if not self._playing:
            self._playing = True
            await self.socket.send_json({"type": "audio_playback_start"})
        max_bytes = int(self.sample_rate * 2 * .20)
        view = memoryview(audio)
        while view and not self._closed:
            async with self._pcm_cond:
                while len(self._pcm) >= max_bytes and not self._closed:
                    await self._pcm_cond.wait()
                n = min(len(view), max_bytes - len(self._pcm))
                self._pcm.extend(view[:n])
                view = view[n:]
                self._pcm_cond.notify_all()

    async def _clear_audio(self):
        async with self._pcm_cond:
            self._pcm.clear()
            self._pcm_cond.notify_all()
        if self._media:
            await self._media.send(bytes([3]))
        self._playing = False

    async def send_json(self, payload):
        typ = payload.get("type")
        if typ in ("answer_start", "answer_interrupt"):
            await self._clear_audio()
        if typ == "answer_done" and self._playing:
            async with self._pcm_cond:
                remainder = len(self._pcm) % self._frame_bytes
                if remainder:
                    self._pcm.extend(bytes(self._frame_bytes - remainder))
                self._pcm_cond.notify_all()

            async def drain():
                while self._pcm and not self._closed:
                    await asyncio.sleep(.02)
                await asyncio.sleep(.15)
                self._playing = False
                if not self._closed:
                    await self.socket.send_json(payload)

            if self._drain_task:
                self._drain_task.cancel()
            self._drain_task = self._task(drain())
            return
        await self.socket.send_json(payload)

    async def close(self):
        self._closed = True
        async with self._pcm_cond:
            self._pcm_cond.notify_all()
        for t in list(self._tasks):
            t.cancel()
        for t in list(self._tasks):
            with suppress(asyncio.CancelledError, Exception):
                await t
        if self._media:
            with suppress(Exception):
                await self._media.close()


async def create_transport(kind, socket, sample_rate=24000):
    if kind == "websocket":
        transport = WebSocketTransport(socket)
    elif kind == "rtc":
        transport = RTCTransport(socket, sample_rate)
    else:
        raise ValueError(f"unknown transport: {kind}")
    await transport.start()
    return transport
