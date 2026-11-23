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
# 国内连不上 huggingface.co 时，可以先 export HF_ENDPOINT=https://hf-mirror.com（第三方镜像）再运行本脚本
HF_HUB_OFFLINE=0 .venv312/bin/python -W ignore -c "
from huggingface_hub import snapshot_download
for r in ['mlx-community/whisper-large-v3-turbo', 'mlx-community/Qwen2.5-7B-Instruct-4bit']:
    snapshot_download(r)
"

# 4. 在本机把 Whisper 压成 4 位量化版（实测准确率、速度不变，GPU 内存 2.1GB -> 1.1GB）
if [ ! -f models/whisper-large-v3-turbo-q4/weights.safetensors ]; then
  echo "== 生成 Whisper 4 位量化版"
  HF_HUB_OFFLINE=1 .venv312/bin/python -W ignore -c "
import json, mlx.core as mx, mlx.nn as nn
from pathlib import Path
from mlx.utils import tree_flatten
from huggingface_hub import snapshot_download
from mlx_whisper.load_models import load_model
src = Path(snapshot_download('mlx-community/whisper-large-v3-turbo'))
m = load_model(str(src), dtype=mx.float16)
nn.quantize(m, group_size=64, bits=4)
out = Path('models/whisper-large-v3-turbo-q4'); out.mkdir(parents=True, exist_ok=True)
mx.save_safetensors(str(out / 'weights.safetensors'), dict(tree_flatten(m.parameters())))
cfg = json.loads((src / 'config.json').read_text()); cfg['quantization'] = {'group_size': 64, 'bits': 4}
(out / 'config.json').write_text(json.dumps(cfg))
"
fi

echo
echo "装好了。启动：.venv312/bin/python server.py  然后用 Chrome 打开 http://localhost:8765"
echo "会议模式还需要虚拟声卡：bash drivers/build_blackhole.sh && sudo bash drivers/install_blackhole.sh"
