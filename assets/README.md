# assets

图片和演示用的音频素材。

## input.wav

README 第一个例子（`vm.ingest(audio="input.wav")`）用的就是它。
内容是「我是素食主义者，对坚果过敏。」，5.8 秒，16 kHz 单声道 PCM16，
OpenAI TTS 合成。末尾留了 1 秒静音——流式那条路要连续 0.5 秒静音才判「说完了」，
音频在说完那一刻就结束的话，`turn_over` 永远等不到。

## question.wav

README 流式那一节喂进 `vm.stream()` 的那段。内容是「我的饮食禁忌是什么？」，
3.2 秒，16 kHz 单声道 PCM16，macOS `say -v Tingting` 合成。

**是个问句**——流式那段演示的就是它：记忆在人还没说完时就查好了，而最后那次
`ingest()` 因为句子里没有关于用户的新事实，会被判成不入库（`facts_count` 0）。
末尾同样留了 1 秒多静音，`turn_over` 才等得到。

## speech.wav

`examples/02_streaming.py` 的默认输入。
内容是「我喜欢吃马卡龙」，7.7 秒，16 kHz 单声道 PCM16。

## cafe_song.wav

「在咖啡馆听到的那首歌」—— web demo 里问起时会把这段原声放回来。

15 秒，16 kHz 单声道 PCM16。由两段素材混成：

| 层 | 来源 | 许可 |
|---|---|---|
| 音乐 | *Chili Pepper* — Fred Longshaw，1927 年爵士钢琴录音（[Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Chili_Pepper_by_Fred_Longshaw_(1927,_Jazz_piano).opus)） | 公有领域（1927 年录音） |
| 环境 | *Restaurant ambience*（[Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Restaurant_ambience.ogg)） | 见 Commons 文件页 |

混音参数（音乐压到背景层，像隔着店里的音响）：

```bash
ffmpeg -ss 8 -t 15 -i music.opus -stream_loop -1 -t 15 -i amb.ogg \
  -filter_complex "[0:a]volume=0.55,highpass=f=120,lowpass=f=6500[m];\
[1:a]volume=1.0[a];[m][a]amix=inputs=2:duration=first:normalize=0,\
dynaudnorm=p=0.7,alimiter=limit=0.95[out]" \
  -map "[out]" -ac 1 -ar 16000 -c:a pcm_s16le cafe_song.wav
```

换成自己的录音：直接覆盖这个文件即可（归档表记的是路径），但**建议重跑一次
ingest** —— `tune:` / `scene:` 标签是从音频算出来的，换了内容标签就对不上了。
