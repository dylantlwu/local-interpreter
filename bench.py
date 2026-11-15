"""测速：用 macOS 自带的 say 合成一句英文和一句中文，走一遍 识别 → 翻译，打印耗时。
运行:  .venv312/bin/python bench.py
"""
import subprocess
import tempfile
import time
from pathlib import Path

import server

SAMPLES = [
    ("en", "en2zh", "Samantha", "Can you walk me through how you would design the database schema for an order matching engine?"),
    ("zh", "zh2en", "Tingting", "我用八个AI工程师，一个月做出了一个交易回放平台，后端是Python，数据库用的是Postgres。"),
]


def synth(text, voice, out_dir):
    aiff = Path(out_dir) / "s.aiff"
    subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(aiff), "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
        check=True, capture_output=True,
    ).stdout
    return raw


if __name__ == "__main__":
    t = time.time()
    server.load_models()
    print(f"加载 {time.time() - t:.1f}s\n")
    with tempfile.TemporaryDirectory() as d:
        for lang, direction, voice, text in SAMPLES:
            pcm = synth(text, voice, d)
            secs = len(pcm) / 4 / 16000
            for run in range(2):
                t0 = time.time()
                heard = server.transcribe(pcm, lang)
                t1 = time.time()
                first = None
                out = ""
                for chunk in server.stream_translation(heard, direction, []):
                    first = first or time.time()
                    out += chunk
                t2 = time.time()
                print(f"[{lang} 第{run + 1}次] 音频 {secs:.1f}s | 识别 {t1 - t0:.2f}s | 翻译首字 {first - t1:.2f}s | 翻译完成 {t2 - t1:.2f}s")
            print(f"  听到: {heard}\n  译文: {out}\n")
