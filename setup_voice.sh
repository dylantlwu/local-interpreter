#!/bin/bash
# 声音克隆的安装（两个引擎各一个 Python 环境，互不干扰）。先跑过 setup.sh（要用它下载的 Whisper 做质检）。
# 用法：bash setup_voice.sh          # 两个都装
#       bash setup_voice.sh f5       # 只装 F5-TTS（像，但权重不能商用，约 1.4GB）
#       bash setup_voice.sh sovits   # 只装 GPT-SoVITS（MIT 可商用，约 6GB）
# 模型都从 ModelScope 下载（国内能直连；Hugging Face 连不上也没关系）。
set -e
cd "$(dirname "$0")"
WHAT=${1:-all}
MS=https://www.modelscope.cn/models

if [ "$(uname -m)" != "arm64" ]; then echo "需要 Apple Silicon（M1 及以后）的 Mac"; exit 1; fi
command -v uv >/dev/null 2>&1 || export PATH="$(python3 -m site --user-base)/bin:$PATH"
command -v uv >/dev/null 2>&1 || { echo "先运行 bash setup.sh"; exit 1; }

get() {  # 断点续传；60 秒低于 20KB/s 就放弃，重跑会接着下
  mkdir -p "$(dirname "$2")"
  [ -s "$2" ] && [ ! -f "$2.part" ] && return 0
  touch "$2.part"
  curl -fL -C - --speed-limit 20000 --speed-time 60 -o "$2" "$1"
  rm -f "$2.part"
}

if [ "$WHAT" = all ] || [ "$WHAT" = f5 ]; then
  echo "== F5-TTS 环境（.venv-clone）"
  uv venv -q -p 3.12 .venv-clone
  # mlx 要钉在 0.25.2：新版的 mx.random.normal 改了参数，f5-tts-mlx 会报错
  uv pip install -q -p .venv-clone/bin/python mlx==0.25.2 f5-tts-mlx==0.2.6 vocos-mlx==0.0.7 \
    mlx-whisper==0.4.3 soundfile pypinyin jieba
  echo "== 下载 F5-TTS v1 权重和 Vocos（约 1.4GB）"
  get $MS/AI-ModelScope/F5-TTS/resolve/master/F5TTS_v1_Base/model_1250000.safetensors models/f5-tts-v1/model_v1.safetensors
  get $MS/AI-ModelScope/F5-TTS/resolve/master/F5TTS_v1_Base/vocab.txt models/f5-tts-v1/vocab.txt
  get $MS/mlx-community/vocos-mel-24khz/resolve/master/config.yaml models/vocos-mel-24khz/config.yaml
  get $MS/mlx-community/vocos-mel-24khz/resolve/master/model.safetensors models/vocos-mel-24khz/model.safetensors
fi

if [ "$WHAT" = all ] || [ "$WHAT" = sovits ]; then
  echo "== GPT-SoVITS 代码（钉在测试过的版本）"
  mkdir -p engines
  if [ ! -d engines/GPT-SoVITS ]; then
    git clone -q https://github.com/RVC-Boss/GPT-SoVITS.git engines/GPT-SoVITS
  fi
  git -C engines/GPT-SoVITS -c advice.detachedHead=false checkout -q 6d51ffceafbf240145619af4c92408a6e54b7924 2>/dev/null \
    || echo "（浅克隆拿不到指定版本，用最新版）"
  echo "== GPT-SoVITS 环境（.venv-sovits）"
  uv venv -q -p 3.12 .venv-sovits
  uv pip install -q -p .venv-sovits/bin/python torch==2.7.0 torchaudio==2.7.0
  uv pip install -q -p .venv-sovits/bin/python -r engines/GPT-SoVITS/requirements.txt
  # resampy：读参考音频要用；setuptools<81：Resemblyzer 还要 pkg_resources
  uv pip install -q -p .venv-sovits/bin/python resampy==0.4.3 "setuptools<81" Resemblyzer==0.1.4 mlx-whisper==0.4.3 soundfile
  echo "== 下载 GPT-SoVITS 预训练模型（约 5GB）"
  P=$MS/XXXXRT/GPT-SoVITS-Pretrained/resolve/master
  if [ ! -f engines/GPT-SoVITS/GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth ]; then
    get $P/pretrained_models.zip engines/dl/pretrained_models.zip
    unzip -q -o engines/dl/pretrained_models.zip -d engines/GPT-SoVITS/GPT_SoVITS
  fi
  if [ ! -d engines/GPT-SoVITS/GPT_SoVITS/text/G2PWModel ]; then
    get $P/G2PWModel.zip engines/dl/G2PWModel.zip
    unzip -q -o engines/dl/G2PWModel.zip -d engines/GPT-SoVITS/GPT_SoVITS/text
  fi
  if [ ! -d engines/nltk/nltk_data ]; then
    get $P/nltk_data.zip engines/dl/nltk_data.zip
    mkdir -p engines/nltk && unzip -q -o engines/dl/nltk_data.zip -d engines/nltk
  fi
  rm -rf engines/dl
fi

echo
echo "装好了。下一步：启动 .venv312/bin/python server.py，用 Chrome 打开 http://localhost:8765/record 录音。"
echo "录完后看 README「声音克隆」一节：先零样本试听，再微调，再念长文。"
