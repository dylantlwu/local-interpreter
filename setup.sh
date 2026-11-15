#!/bin/bash
# 一键安装：Python 环境 + 依赖 + 本地模型（约 7GB，只需一次）
# 用法：bash setup.sh
set -e
cd "$(dirname "$0")"

if [ "$(uname -m)" != "arm64" ]; then echo "需要 Apple Silicon（M1 及以后）的 Mac"; exit 1; fi

# 1. Python 3.12 环境（用 uv，不改动系统自带的 Python）
if ! command -v uv >/dev/null 2>&1; then
  echo "== 安装 uv"
  python3 -m pip install --user --quiet uv
  export PATH="$(python3 -m site --user-base)/bin:$PATH"
fi
echo "== 创建 Python 3.12 环境"
uv python install 3.12
uv venv -q -p 3.12 .venv312
uv pip install -q -p .venv312/bin/python -r requirements.txt

# 2. 不走 Hugging Face 的两个模型
mkdir -p models
if [ ! -f models/kokoro-v1.0.onnx ]; then
  echo "== 下载 Kokoro 朗读模型（约 350MB）"
  curl -fSL -o models/kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
  curl -fSL -o models/voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
fi
ZF=sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
if [ ! -d models/$ZF ]; then
  echo "== 下载中英双语流式识别模型（约 500MB）"
  curl -fSL -o models/zf.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$ZF.tar.bz2
  tar xjf models/zf.tar.bz2 -C models && rm models/zf.tar.bz2
fi

# 3. Hugging Face 上的两个模型（Whisper 1.6GB、Qwen 4GB）
echo "== 下载 Whisper 和 Qwen（约 5.6GB）"
.venv312/bin/python -W ignore -c "
from huggingface_hub import snapshot_download
for r in ['mlx-community/whisper-large-v3-turbo', 'mlx-community/Qwen2.5-7B-Instruct-4bit']:
    snapshot_download(r)
"

echo
echo "装好了。启动：.venv312/bin/python server.py  然后用 Chrome 打开 http://localhost:8765"
echo "会议模式还需要虚拟声卡：bash drivers/build_blackhole.sh && sudo bash drivers/install_blackhole.sh"
