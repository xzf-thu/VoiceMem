class VoiceMemPCMPlayer extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.offset = 0;
    this.bufferedFrames = 0;
    this.started = false;
    this.draining = false;
    this.paused = false;
    this.baseTargetFrames = Math.round(sampleRate * 0.08);
    this.targetFrames = this.baseTargetFrames;
    this.maxTargetFrames = Math.round(sampleRate * 0.32);
    this.stableFrames = 0;
    this.reportFrames = 0;

    this.port.onmessage = (event) => {
      const message = event.data || {};
      if (message.type === "config") {
        const seconds = Math.max(0.04, Math.min(0.32, Number(message.prebuffer) || 0.08));
        this.baseTargetFrames = Math.round(sampleRate * seconds);
        this.targetFrames = Math.max(this.baseTargetFrames, this.targetFrames);
        return;
      }
      if (message.type === "audio" && message.samples) {
        const samples = message.samples;
        if (samples.length) {
          this.queue.push(samples);
          this.bufferedFrames += samples.length;
        }
        return;
      }
      if (message.type === "drain") {
        this.draining = true;
        if (!this.bufferedFrames) this._drained();
        return;
      }
      if (message.type === "pause") {
        this.paused = true;
        this._report("paused");
        return;
      }
      if (message.type === "resume") {
        this.paused = false;
        this._report("resumed");
        return;
      }
      if (message.type === "clear") this._clear(true);
    };
  }

  _clear(notify) {
    this.queue.length = 0;
    this.offset = 0;
    this.bufferedFrames = 0;
    this.started = false;
    this.draining = false;
    this.paused = false;
    this.stableFrames = 0;
    if (notify) this.port.postMessage({ type: "drained", bufferedMs: 0 });
  }

  _drained() {
    this._clear(false);
    this.targetFrames = this.baseTargetFrames;
    this.port.postMessage({ type: "drained", bufferedMs: 0 });
  }

  _report(type) {
    this.port.postMessage({
      type,
      bufferedMs: Math.round((this.bufferedFrames / sampleRate) * 1000),
      targetMs: Math.round((this.targetFrames / sampleRate) * 1000),
    });
  }

  process(_inputs, outputs) {
    const output = outputs[0] && outputs[0][0];
    if (!output) return true;
    output.fill(0);

    // 候选插话期间输出静音并保留队列，resume 后从暂停位置继续。
    if (this.paused) return true;

    if (!this.started) {
      if (!this.bufferedFrames) return true;
      if (!this.draining && this.bufferedFrames < this.targetFrames) return true;
      this.started = true;
      this._report("started");
    }

    let written = 0;
    while (written < output.length && this.queue.length) {
      const head = this.queue[0];
      const count = Math.min(output.length - written, head.length - this.offset);
      output.set(head.subarray(this.offset, this.offset + count), written);
      written += count;
      this.offset += count;
      this.bufferedFrames -= count;
      if (this.offset >= head.length) {
        this.queue.shift();
        this.offset = 0;
      }
    }

    if (written < output.length && !this.bufferedFrames) {
      if (this.draining) {
        this._drained();
      } else {
        // Increase the jitter buffer after an underrun.
        this.started = false;
        this.stableFrames = 0;
        this.targetFrames = Math.min(
          this.maxTargetFrames,
          this.targetFrames + Math.round(sampleRate * 0.02),
        );
        this._report("underflow");
      }
    } else {
      this.stableFrames += written;
      // Reduce the buffer after stable playback.
      if (this.stableFrames >= sampleRate * 5 && this.targetFrames > this.baseTargetFrames) {
        this.targetFrames = Math.max(
          this.baseTargetFrames,
          this.targetFrames - Math.round(sampleRate * 0.01),
        );
        this.stableFrames = 0;
      }
    }

    this.reportFrames += output.length;
    if (this.reportFrames >= sampleRate / 4) {
      this.reportFrames = 0;
      this._report("buffer");
    }
    return true;
  }
}

registerProcessor("voicemem-pcm-player", VoiceMemPCMPlayer);
