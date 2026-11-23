"""把录音整理成 GPT-SoVITS 的训练清单：data/sovits/wavs/*.wav + data/sovits/train.list

用法：.venv-clone/bin/python sovits_prep.py
念的时候即兴改了词的句子（Whisper 听出来比稿子多字）自动去掉。
清单每行：wav路径|说话人|语言|文字。语言 zh = 中英混说（GPT-SoVITS v2 起支持），en = 纯英文。
和 F5 微调用同一套留出句（clone_train.heldout），两边的验证结果能直接比。
"""
import re

import numpy as np
import soundfile as sf

import clone
from clone_train import heldout, items

OUT = clone.HERE / "data" / "sovits"


def adlib(lid, text, wav, cache):
    """念的时候即兴改了词、跟稿子对不上的句子（Whisper 听到的比稿子多 2 个字/词以上），不拿来训练，
    免得模型学到「文字和声音对不上」。结果缓存在 data/sovits/asr.json，第二次跑不用重听。"""
    from clone_read import tokens
    if lid not in cache:
        import mlx_whisper
        r = mlx_whisper.transcribe(str(wav), path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
                                   language="zh" if re.search(r"[\u4e00-\u9fff]", text) else "en",
                                   condition_on_previous_text=False)
        cache[lid] = r["text"].strip()
    heard = cache[lid]
    latin = lambda x: len(re.findall(r"[A-Za-z]+", x))
    if latin(text) and re.search(r"[\u4e00-\u9fff]", text) and latin(heard) < latin(text) / 2:
        return False  # 中英混说的句子 Whisper 常把英文译成中文，这时比不了，保留
    return len(tokens(heard)) > len(tokens(text)) + 1


def main():
    wavs = OUT / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    import json
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    cache_f = OUT / "asr.json"
    cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    rows, held, sec, skipped = [], [], 0.0, []
    for lid, text, path in items():
        free = clone.takes()[lid].get("free")  # 自由讲的文字本来就是照录音校对的，不用查
        a = clone.load_trim(path)
        a = a / (np.abs(a).max() + 1e-9) * 0.9
        f = wavs / f"{lid}.wav"
        sf.write(f, a, clone.SR)
        if not free and adlib(lid, text, f, cache):
            skipped.append(f"  稿：{text}\n  听：{cache[lid]}")
            f.unlink()
            continue
        zh = re.search(r"[一-鿿]", text)
        line = f"{f}|me|{'zh' if zh else 'en'}|{text.strip()}"
        if heldout(lid):
            held.append(line)
        else:
            rows.append(line)
            sec += len(a) / clone.SR
    (OUT / "train.list").write_text("\n".join(rows) + "\n")
    (OUT / "heldout.list").write_text("\n".join(held) + "\n")
    cache_f.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
    if skipped:
        print(f"跳过 {len(skipped)} 句念的和稿子对不上的：\n" + "\n".join(skipped))
    print(f"训练 {len(rows)} 句（{sec / 60:.1f} 分钟），留出 {len(held)} 句 → {OUT}")


if __name__ == "__main__":
    main()
