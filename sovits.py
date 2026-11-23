"""GPT-SoVITS 引擎（MIT 许可，可以商用）：零样本试听 + 给 clone_read.py 用的合成函数。

用法（.venv-sovits 环境）：
  .venv-sovits/bin/python sovits.py                       # 原版模型，零样本，试听句输出到 data/clone/<时间>-sovits/
  .venv-sovits/bin/python sovits.py --gpt X.ckpt --sovits Y.pth   # 用 sovits_train.py 微调出来的权重

和 F5 不同：GPT-SoVITS 先用一个小 GPT 把文字变成「语音 token」（自己决定念多长、哪里停），
再用 SoVITS 把 token 变成声音。所以不用我们估时长，也就不会因为时长给多了而重复句尾。
参考音频必须 3–10 秒，所以每组只用一句。
"""
import argparse
import datetime
import os
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).parent
ENGINE = HERE / "engines" / "GPT-SoVITS"
SR = 32_000
VERSION = "v2ProPlus"

# 参考：一句读稿（3–10 秒），按原文找录音，没录就自动挑同类的
REFS = {k: clone_refs[:1] for k, clone_refs in __import__("clone").REFS.items() if k in ("朗读", "混说", "英文")}


def load(gpt=None, sovits=None, device="cpu"):
    """加载 GPT-SoVITS。Mac 上官方建议用 CPU（MPS 训练出来音质差，推理先也用 CPU 求稳）。"""
    os.environ.setdefault("NLTK_DATA", str(HERE / "engines" / "nltk" / "nltk_data"))
    os.chdir(ENGINE)  # 它的代码里很多相对路径
    for p in (str(ENGINE), str(ENGINE / "GPT_SoVITS")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from TTS_infer_pack.TTS import TTS, TTS_Config
    cfg = TTS_Config({"custom": dict(TTS_Config.default_configs[VERSION], device=device, is_half=False)})
    tts = TTS(cfg)
    if gpt:
        tts.init_t2s_weights(str(gpt))
    if sovits:
        tts.init_vits_weights(str(sovits))
    return tts


def ref_for(name):
    """参考音频 + 文字：读稿原句，去掉头尾静音，存成 wav。"""
    import clone
    e = clone.takes()[clone.pick(REFS[name], name, 1)[0]]
    out = HERE / "data" / "sovits" / "refs"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.wav"
    if not path.exists():
        sf.write(path, clone.load_trim(clone.REC / e["file"]), clone.SR)
    return path, e["text"]


def lang_of(text):
    return "zh" if re.search(r"[一-鿿]", text) else "en"


# 额外参考（音色融合）：平铺直叙的读稿句，多给几段让音色更稳。只影响音色，不影响内容
AUX = ["赚钱的时候不要飘，亏钱的时候不要慌。", "我跟你说，这个市场永远是对的，错的是你自己。",
       "主要是解决现金流的问题，现金流稳了，什么都好说。", "大模型带小模型，这个思路我觉得是对的。",
       "谁有流量谁说了算，这个在哪个行业都一样。", "我减仓了一半，剩下的看情况。"]


def aux_refs(n):
    """n 段额外参考的 wav 路径（去掉头尾静音存好）。没录的句子自动挑同类的补上。"""
    import clone
    if not n:
        return []
    t, out = clone.takes(), HERE / "data" / "sovits" / "refs"
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for lid in clone.pick(AUX, "中文", n):
        f = out / f"aux_{lid}.wav"
        if not f.exists():
            sf.write(f, clone.load_trim(clone.REC / t[lid]["file"]), clone.SR)
        paths.append(str(f))
    return paths


def say(tts, text, ref_path, ref_text, seed=7, speed=1.0, temperature=1.0, split="cut0", aux=()):
    """念一段话，返回 32kHz 的 numpy 数组。默认不再切（cut0）：clone_read 已经切成 45 字以内，
    再按逗号切，文言文会碎成四字一段，GPT-SoVITS 念短片段很不稳（漏后半句、出空白）。"""
    req = dict(text=text, text_lang=lang_of(text), ref_audio_path=str(ref_path), prompt_text=ref_text,
               prompt_lang=lang_of(ref_text), aux_ref_audio_paths=list(aux), top_k=15, top_p=1.0, temperature=temperature,
               text_split_method=split, batch_size=1, speed_factor=speed, seed=seed,
               parallel_infer=False, repetition_penalty=1.35, fragment_interval=0.3, return_fragment=False)
    chunks = []
    for sr, audio in tts.run(req):
        chunks.append(audio)
    a = np.concatenate(chunks).astype(np.float32)
    return a / 32768.0 if np.abs(a).max() > 2 else a


def main():
    import clone
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", action="append")
    ap.add_argument("--ref", choices=list(REFS), action="append")
    ap.add_argument("--gpt", default="")
    ap.add_argument("--sovits", default="")
    args = ap.parse_args()
    tag = "sovits" + ("-ft" if args.gpt or args.sovits else "")
    out = clone.OUT / (datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{tag}")
    out.mkdir(parents=True)
    tts = load(args.gpt or None, args.sovits or None)
    rows = []
    for name in args.ref or list(REFS):
        ref_path, ref_text = ref_for(name)
        sf.write(out / f"ref_{name}.wav", sf.read(ref_path)[0], clone.SR)
        print(f"\n== 参考「{name}」：{ref_text}")
        for i, text in enumerate(args.text or clone.TESTS):
            t0 = datetime.datetime.now()
            a = say(tts, text, ref_path, ref_text)
            sf.write(out / f"{name}_{i}.wav", a, SR)
            sec = (datetime.datetime.now() - t0).total_seconds()
            rows.append((name, f"ref_{name}.wav", text, f"{name}_{i}.wav", sec))
            print(f"  {sec:.1f}s（音频 {len(a) / SR:.1f}s）  {text}", flush=True)
    clone.write_page(out, rows)
    page = (out / "index.html").read_text().replace("零样本，F5-TTS", f"GPT-SoVITS {VERSION}" + ("，微调" if "ft" in tag else "，零样本"))
    (out / "index.html").write_text(page)
    print(f"\n试听页：{out / 'index.html'}")


if __name__ == "__main__":
    main()
