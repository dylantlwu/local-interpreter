"""比较 GPT-SoVITS 不同训练轮数的权重：同样几段中文、英文，看质检和声纹相似度，挑最好的一轮。

用法（.venv-sovits 环境）：
  .venv-sovits/bin/python sovits_eval.py                  # 评 SoVITS_weights_v2ProPlus/ 下所有 me_e*.pth
  .venv-sovits/bin/python sovits_eval.py --only me_e8 me_e24

声纹相似度：说话人识别模型（ERes2NetV2）算的声纹，和你训练录音的平均声纹比余弦相似度；
你本人没参与训练的录音约 0.90。注意 v2ProPlus 内部也用这个模型，所以数字对 GPT-SoVITS 可能略偏高。
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import sovits  # noqa: E402
from clone_read import chunks, score  # noqa: E402

ZH = [0, 4, 10, 16, 22, 28]  # 《与妻书》里的几段，各种句式都有


def cases():
    zh = chunks((HERE / "examples/yuqishu.txt").read_text())
    en = [p for p, _ in chunks((HERE / "examples/odyssey_book1.txt").read_text()[:6000]) if len(p.split()) > 15][:3]
    return [("朗读", zh[i][0]) for i in ZH] + [("英文", t) for t in en]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--aux", type=int, nargs="*", default=[0], help="额外参考段数，可给多个值逐一比较")
    args = ap.parse_args()
    wdir = sovits.ENGINE / "SoVITS_weights_v2ProPlus"
    ckpts = sorted(wdir.glob("me_e*.pth"), key=lambda p: int(p.stem.split("_e")[1].split("_")[0]))
    if args.only:
        ckpts = [c for c in ckpts if c.stem in args.only]
    tts = sovits.load()
    from sv import SV  # sovits.load 之后才能 import（路径）
    import librosa
    import torch
    sv = SV("cpu", False)

    def emb(a16):
        e = sv.compute_embedding3(torch.tensor(a16)[None]).squeeze(0).numpy()
        return e / np.linalg.norm(e)

    train = [l.split("|")[0] for l in (HERE / "data/sovits/train.list").read_text().split("\n") if l]
    c = np.mean([emb(librosa.load(f, sr=16000)[0]) for f in train], 0)
    c /= np.linalg.norm(c)
    out = HERE / "data/sovits/eval"
    out.mkdir(exist_ok=True)
    refs = {n: sovits.ref_for(n) for n in ("朗读", "英文")}
    results = {}
    for ck, naux in [(c_, n) for c_ in ckpts for n in args.aux]:
        tts.init_vits_weights(str(ck))
        rows = []
        name = ck.stem + (f"+aux{naux}" if naux else "")
        for i, (ref, t) in enumerate(cases()):
            a = sovits.say(tts, t, *refs[ref], seed=7, speed=0.9, aux=sovits.aux_refs(naux))
            f = out / f"{name}_{i}.wav"
            sf.write(f, a, sovits.SR)
            sc, _ = score(t, f)
            sim = float(emb(librosa.resample(a, orig_sr=sovits.SR, target_sr=16000)) @ c)
            rows.append(dict(lang="zh" if ref == "朗读" else "en", score=sc, sim=sim))
        results[name] = rows
        for lang in ("zh", "en"):
            r = [x for x in rows if x["lang"] == lang]
            print(f"{name:18s} {lang}  质检 {np.mean([x['score'] for x in r]):.0%}"
                  f"（过 {sum(x['score'] >= 0.85 for x in r)}/{len(r)}）  声纹 {np.mean([x['sim'] for x in r]):.3f}", flush=True)
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
