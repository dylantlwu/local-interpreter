# 本地同传 · Local Interpreter

**在你自己的 Mac 上跑的中英同声传译。不联网，不上云，数据不出电脑。**
Real-time Chinese ↔ English interpretation that runs entirely on an Apple Silicon Mac.

- **现场模式**：一个麦克风，直接说话，自动分辨中文还是英文，屏幕上出译文，可以用自然的英文人声读出来。
- **会议模式**：夹在你和会议软件（飞书、Zoom 等）之间——对方的英文变成你屏幕上的中文字幕，你说中文，对方听到英文语音。两路声音分开，可以同时说话。
- **边听边翻**：说话过程中 0.5 秒左右出实时字幕，1 秒多出淡色的草稿译文（结尾可预测时会提前译出，用虚线下划线标注），说完后换成精确结果。

> 实测环境：M4 Mac mini，16GB 内存。飞书会议模式已打通；微信等跟随系统音频设备的软件仍在测试。

## 快速开始

需要：Apple Silicon Mac（M1 及以后，建议 16GB 内存）、Chrome、约 8GB 磁盘空间。

```bash
git clone https://github.com/dylantlwu/local-interpreter.git
cd local-interpreter
bash setup.sh                    # 装 Python 环境、依赖和模型（约 7GB，只需一次）
.venv312/bin/python server.py    # 等终端出现「模型已加载」（约 15 秒）
```

用 Chrome 打开 http://localhost:8765 ，点「开始」，允许麦克风。

## 它是怎么工作的

两遍识别（借鉴 FunASR 2pass 和 DeepL 的稳定文本流）：

| 阶段 | 做什么 | 模型 | 跑在 |
|---|---|---|---|
| 说话中 | 每 0.3 秒出实时字幕、判断中英文 | sherpa-onnx 双语流式 zipformer | CPU（约占实时 3%） |
| 说话中 | 草稿翻译，最多每秒一次；结尾好猜时一起译出（虚线下划线） | Qwen2.5-7B（提示词 KV 缓存，首字约 0.2 秒） | GPU，空闲时才做 |
| 说话中 | 停顿时判断"说完没有"，有把握就提前切句 | Qwen2.5-7B（看下一个字是句号的概率） | GPU |
| 说完后 | 精确识别 → 正式翻译，替换草稿；正式任务一来草稿立刻让出 GPU | Whisper large-v3-turbo → Qwen2.5-7B | GPU |
| 朗读 | 英文译文按句合成，边合成边播 | Kokoro | CPU |

实测（M4 / 16GB）：一句话说完后约 2 秒出正式译文首字；朗读在第一句译完后约 0.5–1 秒出声。

## 会议模式（飞书 / Zoom 等）

```
对方 ─▶ 会议软件扬声器 = BlackHole 2ch ─▶ 本地同传 ─▶ 中文字幕 + 原声转到耳机
你   ─▶ 真麦克风 ─▶ 本地同传 ─▶ 英文语音 ─▶ BlackHole 16ch = 会议软件麦克风 ─▶ 对方
```

**1. 装两个虚拟声卡（一次）**：从 [BlackHole](https://github.com/ExistentialAudio/BlackHole) 开源代码在本机编译，只需要 Xcode 命令行工具：

```bash
bash drivers/build_blackhole.sh
sudo bash drivers/install_blackhole.sh
```

卸载：`sudo rm -rf /Library/Audio/Plug-Ins/HAL/BlackHole*ch.driver && sudo killall coreaudiod`

**2. 会议软件的音频设置**：麦克风选 **BlackHole 16ch**，扬声器选 **BlackHole 2ch**。
飞书里一定要在列表里**直接点选具体设备**，选"与系统一致"实测读不到声音。

**3. 本地同传页面**选「会议」模式，点开始，然后选：我的麦克风 = 真麦克风；会议声音来自 = BlackHole 2ch；英文语音送到 = BlackHole 16ch；会议原声播放到 = 你的耳机。

**跟随系统音频设备的软件（比如 Mac 版微信）**：通话前把系统设置 → 声音里的输出改成 BlackHole 2ch、输入改成 BlackHole 16ch，"播放声音效果所用设备"也改成 BlackHole 2ch，并在改完之后重启该软件；通话结束后改回来。

**注意**
- 一定要戴耳机，否则你的麦克风会收进对方的声音。
- 对方只听到英文合成语音，比你开口晚约 2–3 秒（同传本身的延迟）。

## 用到的模型和许可

| 用途 | 模型 / 组件 | 大小 | 许可 |
|---|---|---|---|
| 精确识别 | [mlx-community/whisper-large-v3-turbo](https://huggingface.co/mlx-community/whisper-large-v3-turbo) | 1.6 GB | MIT |
| 翻译 | [mlx-community/Qwen2.5-7B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit) | 4.0 GB | Apache-2.0 |
| 朗读 | [Kokoro v1.0](https://github.com/thewh1teagle/kokoro-onnx) | 0.35 GB | Apache-2.0 |
| 实时字幕 | [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) streaming zipformer bilingual zh-en | 0.2 GB | Apache-2.0 |
| 虚拟声卡 | [BlackHole](https://github.com/ExistentialAudio/BlackHole)（本机从源码编译，不随本仓库分发） | — | GPL-3.0 |

本仓库代码使用 MIT 许可。环境变量 `ASR_MODEL`、`MT_MODEL` 可换模型，`VOICE` 设默认朗读声音。

## 已知局限

- 实时字幕（流式模型）会听错词，草稿译文因此比较粗糙；正式结果不受影响。
- 7B 翻译模型偶尔会在草稿里自己补全句子。
- 三个模型约占 6GB 内存，16GB 的 Mac 同时开会议软件和浏览器时比较紧张；服务空闲时每 20 秒预热一次，避免模型被换到磁盘后卡顿。
- 现场模式朗读期间不收音（避免把朗读当成对方说话）；会议模式没有这个限制。

## 其他

- 只看界面、不加载模型：`MOCK=1 .venv312/bin/python server.py`
- 测速：`.venv312/bin/python bench.py`（会再加载一份模型，别和服务同时跑）
- 可调参数在 `index.html` 的 `VAD` 里：停顿多久算一句结束（`endSilenceMs`，默认 700ms）、长句在换气处切（`softMaxMs`，默认 7 秒）等。
