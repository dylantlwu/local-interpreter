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

**会议中切换对方听到什么**（会议软件的麦克风一直选 BlackHole 16ch，不用再去会议软件里切）：页面上的按钮或数字键

| 键 | 模式 | 对方听到 |
|---|---|---|
| 1 | 翻译 | 英文语音 |
| 2 | 直通 | 你的原声（想直接说英文时） |
| 3 | 混合 | 小音量原声 + 英文语音 |
| 4 | 静音 | 什么都没有 |

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
| 声音克隆（试验） | [F5-TTS v1 Base](https://github.com/SWivid/F5-TTS)（权重从 ModelScope `AI-ModelScope/F5-TTS` 下载）+ [Vocos](https://github.com/gemelo-ai/vocos)，用 [f5-tts-mlx](https://github.com/lucasnewman/f5-tts-mlx) 在本地跑 | 1.4 GB | 代码 MIT；**权重 CC-BY-NC-4.0，只能非商用** |
| 虚拟声卡 | [BlackHole](https://github.com/ExistentialAudio/BlackHole)（本机从源码编译，不随本仓库分发） | — | GPL-3.0 |

本仓库代码使用 MIT 许可。环境变量 `ASR_MODEL`、`MT_MODEL` 可换模型，`VOICE` 设默认朗读声音。

**省内存**：识别默认用本地生成的 Whisper 4 位量化版（实测准确率、速度不变，GPU 内存 2.1GB → 1.1GB）。翻译还可以换成 `MT_MODEL=mlx-community/Qwen3-4B-Instruct-2507-4bit`：再省约 1.9GB、首字快约 0.2 秒，口语更自然，但更容易自己编意思，草稿会擅自补全句子，所以默认仍用 Qwen2.5-7B。服务默认离线运行（`HF_HUB_OFFLINE=1`），不会每次启动都联网检查模型更新。

## 热词表（hotwords.txt）

改完保存即生效，不用重启。三种写法：

```
AI                                   # 识别时优先认出这个词（专有名词、产品名、人名）
同声传逸 => 同声传译                   # 识别常错的写法，自动改正（实时字幕和正式结果都改）
同声传译 = simultaneous interpretation  # 固定译法，翻译时一律这样译
```

## 场景（scenes/ 文件夹）

页面上的「场景」下拉菜单切换。每个场景一个文件，写法和 hotwords.txt 一样，叠加在通用热词表之上，另外可以写 `@语气: ……` 告诉翻译模型这次对话的语气。自带四个：商务会议、家人朋友、技术交流、交易。新建一个 `.txt` 文件就是新场景，保存即生效。

```
@语气: Trading desk conversation. Use standard market terminology.
爆仓 = liquidation
盘口 = order book
```

## 翻译规则（针对会议里实测出错最多的口语）

- 称呼不是主语：句首的"老婆、宝贝、亲爱的……"是在叫对方（"老婆你听得到吗" → *Honey, can you hear me?*）
- 不换主语："我"就是 I
- "喂喂喂"只译一次，"那个什么"这类填充词自然处理，俗语按意思译
- 识别的同音错字由模型推测本意；一律译成另一种语言，不原样照抄
- 上下文用"原文 → 译文"成对的最近 4 句；很短的句子不带上下文，免得把上一句复述出来

## 杂音和回声怎么过滤

| 源头 | 处理 |
|---|---|
| 铃声、音乐、键盘、呼吸被当成说话 | 流式识别一个字都没听出来（1 秒以上）就不送去精确识别；服务端再用 Silero VAD 检查，人声不到 0.3 秒直接丢掉（约 10ms，不占 GPU） |
| 回声循环（对方外放的英文被你的麦克风收进来 → 再翻译再朗读） | 英文语音在播、以及播完 1.5 秒内不收你的麦克风；会议模式下，"我的麦克风"这一路听出英文、"会议声音"这一路听出中文，都当回声丢掉 |
| Whisper 把识别提示词当结果吐出来 | 英文不用提示词；中文用一句没人会说的话，结果和它高度相似就丢掉 |
| Whisper 幻觉（多语种乱码、"欢迎订阅"、重复循环） | 置信度/压缩率门槛；出现俄文、韩文、日文假名等文字直接丢掉；常见字幕幻觉句式删掉；连续重复只留一次 |

## 会后纠错 → 自动评测 → LoRA 微调

每句正式翻译都会自动记到 `data/history/日期.jsonl`（只在本机，不进仓库）。会后打开 http://localhost:8765/review （同传页面右上角「会后纠错」）：

- 逐句看：原文听错了就改原文，译错了就改译文，<kbd>⌘</kbd>+<kbd>Enter</kbd> 保存（没改就记为"译得对"），<kbd>⌘</kbd>+<kbd>.</kbd> 跳过杂音句。
- 改原文时会建议一条热词改正（例如 `同声专业 => 同声传译`），点一下就加进热词表，以后识别自动改。

攒下来的正确答案有两个用处：

```bash
.venv312/bin/python corrections.py stats      # 攒了多少句
.venv312/bin/python corrections.py eval       # 用正在运行的服务重译这些句子，和正确答案比（chrF 分数 + 完全一致比例）
.venv312/bin/python corrections.py export     # 导出 LoRA 训练集到 data/lora/
```

换模型、改提示词、加场景之后跑一次 `eval`，就知道是变好还是变差，结果存在 `data/evals/`。
**大模型带小模型**：`teacher/cases.tsv` 是大模型（Claude）当老师写的约 200 条教学案例，按本地模型最常犯的错分类：称呼、人称、填充词、口语、识别同音错字、半句话、不加内容、术语、面试问题。`export` 默认用老师案例训练、真实会议句子只用来考试，两边不重复。
首次实测（7B，300 步取第 200 步）：当时译错的 25 句 chrF 44.5 → 59.3，16 句更接近正确答案；但"我老婆（叫人）→ my wife"、"不是吧"、"上架"这几类还没学会，需要更多案例。

句子攒到几百句后可以做 LoRA：`export` 会打印训练命令（先停服务省内存），训练完用 `MT_ADAPTER=data/adapters` 启动服务，再跑 `eval --heldout` 只看没训练过的那 1/10，避免拿练过的题考自己。

## 声音克隆：用你自己的声音念信、念书（快速上手）

录 15–20 分钟自己的声音，在本机微调一个「你的声音」，然后让它念任何文字（信、书、睡前故事）。录音、模型、生成的声音都只在你的电脑上。

> **只克隆你自己、或者明确同意的人的声音。** 生成的声音要标明是 AI 生成（试听页已经自动标了）。不要用来冒充别人。

**1. 安装**（先按上面「快速开始」跑过 `setup.sh`）

```bash
bash setup_voice.sh          # 两个引擎都装（约 8GB）；只要一个：bash setup_voice.sh f5 或 sovits
```

| 引擎 | 像不像（实测） | 速度 | 许可 |
|---|---|---|---|
| F5-TTS（`.venv-clone`） | 更像 | 念 7 分钟要约 24 分钟 | 权重 CC-BY-NC，**只能非商用** |
| GPT-SoVITS（`.venv-sovits`） | 稍差 | 念 7 分钟约 4 分钟 | MIT，可商用 |

**2. 录音**：启动服务（`.venv312/bin/python server.py`），Chrome 打开 http://localhost:8765/record ，照着稿子一句一句读。

- 1–9 段是聊天口吻（约 8 分钟），10 段是自由讲的回忆问题（可跳过），11–15 段是「念给人听」的朗读口吻（约 10 分钟）。想念信念书，11–15 段一定要录。
- 同一个房间、同一支麦克风，离嘴一拳远；读错了按 R 重录这一句。
- 想换成自己的稿子：复制 `recording/script.txt` 为 `recording/script.local.txt` 再改（这个文件不进 git，有它就优先用它）。

**3. 先听零样本**（不训练，直接拿几句录音当参考）

```bash
.venv-clone/bin/python clone.py                  # 结果在 data/clone/<时间>/index.html
```

**4. 微调**（让它更像你；训练期间最好关掉同传服务省内存）

```bash
.venv-clone/bin/python clone_train.py --steps 1200 --save-every 300      # F5，M4 约 17 分钟
.venv-clone/bin/python sovits_prep.py && .venv-sovits/bin/python sovits_train.py   # GPT-SoVITS，16 轮约 1 小时
```

**5. 念长文**（`examples/` 里有《与妻书》《诫子书》《奥德赛》第一卷，都是公版）

```bash
.venv-clone/bin/python clone_read.py examples/yuqishu.txt --ref 朗读 --ckpt data/clone_ft/step_1200.safetensors
.venv-sovits/bin/python clone_read.py examples/yuqishu.txt --engine sovits --ref 朗读 --aux 6 \
    --sovits auto            # auto = 用训练轮数最多的权重
```

输出在 `data/longform/out/…/index.html`：整段音频 + 原文，每段都用 Whisper 自动质检，不过关的会自动重念。用自己的文字就换成任意 `.txt`（空行分段）。

下面是做这些功能时的详细说明和实测记录（数字都是原作者自己的声音）。

## 保存我的录音（为声音克隆和个人模型攒数据）

页面底部勾上「保存我的录音」（默认关闭，设置会记住）：我说的每句话存一段 16kHz wav 到 `data/voice/日期/`，文字和时长记在 `data/voice/日期.jsonl`，只在本机。会议模式只存「我的麦克风」这一路；现场模式会存下所有说话声，别人的声音在纠错页面跳过。

**读稿录音**（http://localhost:8765/record ，同传页面右上角「读稿录音」）：照着 `recording/script.txt` 一句一句读，录音和正确文字直接配对，不用再校对。稿子约 10 分钟、156 句，其中 65 句中英夹杂（插词、英文动词加中文语法、整句切换、缩写念法、中英语气词……），最后一段是 20 个回忆访谈问题（自由讲，录完自动识别成文字，在页面上改对）。录的是 48kHz 原声（关掉浏览器的降噪、回声消除、自动增益），存在 `data/recording/`；每句会检查爆音、太小声、没听到人声。键盘：空格开始/停止，P 回放，R 重录，←→ 换句。稿子可以直接改，改过的句子算新的一句。

纠错页面里有录音的句子可以直接播放（<kbd>⌘</kbd>+<kbd>P</kbd>），照着录音把原文改成一字不差：

```bash
.venv312/bin/python corrections.py voice            # 攒了多少分钟、校对了多少
.venv312/bin/python corrections.py voice --export   # 校对过的录音 + 文字 → data/voice_dataset/（wavs/ + metadata.csv）
```

### 声音克隆试听（零样本）

拿几句读稿录音做参考（约 10 秒，文字就是稿子原文），让 F5-TTS 用你的声音念新的话，全部在本地跑：

```bash
.venv-clone/bin/python clone.py                          # 混说 / 中文 / 英文三组参考 × 5 句试听句
.venv-clone/bin/python clone.py --ref 混说 --text "要念的话"
```

结果在 `data/clone/<时间>/index.html`，真人原声和克隆放在一起对比听。M4 上念 7 秒的话约 40 秒。克隆用单独的环境 `.venv-clone`（`bash setup_voice.sh f5` 安装；mlx 钉在 0.25.2，新版不兼容），模型放在 `models/f5-tts-v1/`、`models/vocos-mel-24khz/`。F5-TTS 的权重是非商用许可，做成收费服务前要换成可商用的模型或自己训练。

### 微调：让克隆更像你

```bash
.venv-clone/bin/python clone_train.py --steps 1200 --save-every 300   # 读稿录音训练，M4 约 17 分钟，内存峰值约 7GB
.venv-clone/bin/python clone.py --ckpt data/clone_ft/step_1200.safetensors
```

用读稿录音（文字已知）加校对过的自由讲做训练数据，留出约 8% 的句子算验证损失（随机数固定，各存档点之间能直接比）。为了在 16G 内存的机器上跑，前 6 层冻结、超过 10 秒的句子不训练、显存缓存上限 1G。第一次 1200 步：验证损失 0.647 → 0.582，没有过拟合。注意 mlx 存档遇到中文路径会段错误，脚本里已经绕开。

### 长篇朗读：书、信

```bash
.venv-clone/bin/python clone_read.py examples/yuqishu.txt --ref 中文 --ckpt data/clone_ft/step_1200.safetensors
.venv-clone/bin/python clone_read.py examples/odyssey_book1.txt --ref 英文 --ckpt ... --max-words 3000   # 英文约 150 词/分钟
```

按句子切成 20 秒以内的小段（中文 45 字、英文 40 词，太长的句子再按逗号切），逐段生成后拼成 `full.wav`，附带一个边听边看原文的页面。**每段自动质检**：用 Whisper 听一遍，按无声调拼音（中文）或单词（英文）和原文比对，同音字不算错，漏念和重复都扣分；低于 85% 就换随机种子重念，最多 3 次，保留最好的一次，仍不过的段在页面上标「可能念得不准」。断了重跑会接着念；`--recheck` 会把旧的段重新质检。异体字（畄、廽、厛、亗）要先换成通行字，否则拼音会出错。

实测《与妻书》（1911，约 1200 字，7 分钟）：不切短、不质检时按汉字比对只对上 54%（不少是识别模型把古文听成同音字），但也有一段整段念成同一个字、开头一句被跳过；加质检重念后 31 段过了 29 段。没过的段里出现过参考录音里的话（「半个小时」），说明模型偶尔会把参考音频的内容念出来。

**重复念句尾的原因和修法**：F5 不自己决定念多长，时长是我们给的。原来按参考那两句估，给成 2.5 字/秒，比你平时（读稿录音实测 3.45 字/秒）慢太多，多出来的时间模型就拿去重复句尾。现在按你自己的语速算（念书默认 `--speed 0.9`，约 3.1 字/秒），参考换成平铺直叙的「朗读」组（聊天口吻的「中文」组会把「断掉了」「半个小时」串进去），质检也会抓重复（原文没有、却出现两次的连续 4 个字/词，直接判不过）。改完《与妻书》31/31 通过，没有重复。

### 换成 GPT-SoVITS（MIT，可商用）

F5 的权重不能商用，所以加了第二个引擎 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)（v2ProPlus，代码装在 `engines/GPT-SoVITS`，环境 `.venv-sovits`，模型从 ModelScope `XXXXRT/GPT-SoVITS-Pretrained` 下载）。它先用一个小 GPT 把文字变成语音 token（自己决定念多长），再用 SoVITS 变成声音，所以不用估时长。

```bash
.venv-clone/bin/python sovits_prep.py                 # 录音 → 训练清单（Whisper 自动去掉念的时候即兴改词的句子）
.venv-sovits/bin/python sovits_train.py               # 预处理 + 微调（Mac 上用 CPU，SoVITS 16 轮约 1 小时）
.venv-sovits/bin/python sovits.py                     # 零样本试听
.venv-sovits/bin/python clone_read.py examples/yuqishu.txt --engine sovits --ref 朗读 \
    --sovits auto
```

实测《与妻书》两个引擎对比（声纹相似度 = 说话人识别模型算的余弦相似度，和你训练录音的平均声纹比；你本人没参与训练的录音是 0.904）：

| | 质检通过 | 平均质检 | 声纹相似度 | 生成时间 |
|---|---|---|---|---|
| F5 微调（step_1200） | 31/31 | 97% | **0.883** | 24 分钟 |
| GPT-SoVITS 微调 SoVITS + 原版 GPT | 30/31 | 96% | 0.835 | **4 分钟** |
| 同上，SoVITS 训到 16 轮（`me_e16_s576.pth`） | 30/31 | — | 0.847 | 4 分钟 |
| 同上 + 6 段参考融合（`--aux 6`） | 30/31 | — | 0.863 | 4 分钟 |
| GPT-SoVITS 零样本（试听句） | — | — | 0.662 | — |

多训几轮帮助不大：8/12/16/20/24 轮用 `sovits_eval.py` 在同样 6 段中文、3 段英文上比，中文声纹 0.834 → 最高 0.854（16 轮）后回落，英文没变好，质检都差不多。**多段参考融合有用**：除了主参考，再给 6 段平铺直叙的读稿句（`clone_read.py --aux 6`，列表在 `sovits.AUX`），模型会把几段的音色平均。因为 ERes2NetV2 正是 v2ProPlus 内部融合用的模型，用它打分有「拿答案对答案」之嫌，所以另用和两个引擎都无关的 Resemblyzer 复核（`voice_sim.py`）：

| 与妻书 | ERes2NetV2 | Resemblyzer（独立） |
|---|---|---|
| 你本人（没参与训练） | 0.904 | 0.917 |
| F5 微调 | 0.883 | 0.881 |
| GPT-SoVITS 16 轮 | 0.847 | 0.830 |
| GPT-SoVITS 16 轮 + 6 段融合 | 0.863 | 0.845 |

两个裁判结论一致：融合能缩小差距，但 F5 仍然明显更像。要再接近 F5，得换别的办法（更多「念书口吻」的录音、多段参考音频融合），不是多训。

踩过的坑：GPT-SoVITS 按逗号切句时，文言文碎成四字一段，会漏后半句、出空白（Whisper 在空白上「听到」了「请不吝点赞订阅」），所以 `sovits.say` 默认不再切（clone_read 已经切到 45 字以内）。微调 GPT（15 轮）反而更容易漏字，目前只用微调的 SoVITS。参考音频要 3–10 秒、不要有缩写（「ASR、TTS」会让它出乱码）。声纹相似度用的模型也是 v2ProPlus 内部用的那个，对 GPT-SoVITS 可能略有偏袒。

## 已知局限

- 实时字幕（流式模型）会听错词，草稿译文因此比较粗糙；正式结果不受影响。
- 7B 翻译模型偶尔会在草稿里自己补全句子。
- 三个模型约占 6GB 内存，16GB 的 Mac 同时开会议软件和浏览器时比较紧张；服务空闲时每 20 秒预热一次，避免模型被换到磁盘后卡顿。
- 现场模式朗读期间不收音（避免把朗读当成对方说话）；会议模式没有这个限制。

## 其他

- 只看界面、不加载模型：`MOCK=1 .venv312/bin/python server.py`
- 测速：`.venv312/bin/python bench.py`（会再加载一份模型，别和服务同时跑）
- 可调参数在 `index.html` 的 `VAD` 里：停顿多久算一句结束（`endSilenceMs`，默认 700ms）、长句在换气处切（`softMaxMs`，默认 7 秒）等。
