"""像不像你：两个独立的说话人识别模型分别打分，和你训练录音的平均声纹比余弦相似度。

用法（.venv-sovits 环境）：.venv-sovits/bin/python voice_sim.py
  - ERes2NetV2：GPT-SoVITS v2ProPlus 内部也用它，所以对 GPT-SoVITS（尤其是多段参考融合）偏松
  - Resemblyzer（GE2E）：和两个引擎都无关，作为独立裁判
基准：你本人没参与训练的 14 句录音。
"""
import glob
import os
import sys
from pathlib import Path

import numpy as np

H = Path(__file__).parent.resolve()
E = H / "engines" / "GPT-SoVITS"
O = H / "data" / "longform" / "out"

VERSIONS = {
    "F5 微调": "yuqishu-朗读-step_1200",
    "GPT-SoVITS 16 轮": "yuqishu-朗读-sovits-me_e16_s576",
    "GPT-SoVITS 16 轮 + 6 段参考融合": "yuqishu-朗读-sovits-me_e16_s576-aux6",
}


def main():
    import librosa
    import torch
    from resemblyzer import VoiceEncoder, preprocess_wav
    os.chdir(E)
    sys.path[:0] = [str(E), str(E / "GPT_SoVITS")]
    from sv import SV
    sv, ge2e = SV("cpu", False), VoiceEncoder("cpu", verbose=False)

    def embs(f):
        a = librosa.load(f, sr=16000)[0]
        e1 = sv.compute_embedding3(torch.tensor(a)[None]).squeeze(0).numpy()
        e2 = ge2e.embed_utterance(preprocess_wav(a, source_sr=16000))
        return e1 / np.linalg.norm(e1), e2 / np.linalg.norm(e2)

    lst = lambda n: [l.split("|")[0] for l in (H / f"data/sovits/{n}.list").read_text().split("\n") if l]
    tr = [embs(f) for f in lst("train")]
    cs = [np.mean([t[k] for t in tr], 0) for k in (0, 1)]
    cs = [c / np.linalg.norm(c) for c in cs]
    rows = [("你本人（没参与训练）", lst("heldout"))] + [
        (name, sorted(glob.glob(str(O / d / "seg" / "0*.wav")))) for name, d in VERSIONS.items()]
    print(f"{'':32s}{'ERes2NetV2':>12s}{'Resemblyzer':>13s}")
    for name, files in rows:
        if not files:
            continue
        e = [embs(f) for f in files]
        s = [np.mean([x[k] @ cs[k] for x in e]) for k in (0, 1)]
        print(f"{name:32s}{s[0]:12.3f}{s[1]:13.3f}   （{len(files)} 段）")


if __name__ == "__main__":
    main()
