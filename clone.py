"""零样本声音克隆试听：拿读稿录音做参考，用 F5-TTS（MLX，本地跑）合成你的声音。

用法（要用 .venv-clone 这个 Python 3.12 环境）：
  .venv-clone/bin/python clone.py                 # 用默认参考和试听句，输出到 data/clone/<时间>/
  .venv-clone/bin/python clone.py --text "要念的话"  # 只念这一句

参考音频 = 几句读稿录音拼起来（去掉前后静音，约 10 秒），文字就是稿子原文，不用识别。
每次生成都写一个 index.html，把「真人原声」和「克隆」放在一起对比听。
"""
import argparse
import datetime
import html
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).parent
REC = HERE / "data" / "recording"
OUT = HERE / "data" / "clone"
SR = 24_000

# 参考：每组两句读稿（按稿子原文找录音），拼成约 10 秒。没录这两句的话自动挑同类的句子
REFS = {
    "混说": ["它其实是有四五个 model 串在一起的，ASR、翻译、TTS，全部在本地跑。",
           "Claude 帮我把整个 pipeline 搭起来，我只负责讲需求。"],
    "中文": ["刚刚断掉了，你最后那句我没听清楚，可以再讲一次吗？", "今天大概讲半个小时，后面我还有个 meeting。"],
    "英文": ["Everything runs locally on my Mac. Nothing goes to the cloud.",
           "We're training a small model with help from a bigger one."],
    # 念书念信用：平铺直叙的两句。「中文」那组是聊天口吻，长篇里会把「断掉了」「半个小时」串进去
    "朗读": ["同声传译最难的不是翻译，是知道对方什么时候讲完。", "赚钱的时候不要飘，亏钱的时候不要慌。"],
}


def line_id(text):
    """录音页给每句的 id = 文字的 sha1 前 10 位（和 server.py 的 script_lines 一致）。"""
    import hashlib
    return hashlib.sha1(text.strip().encode()).hexdigest()[:10]


def script_sections():
    """录音稿里每句属于哪一段：{原文: 段落标题}（私人版 script.local.txt 优先）。"""
    out, sec = {}, ""
    for f in (HERE / "recording" / "script.local.txt", HERE / "recording" / "script.txt"):
        if f.exists():
            for line in f.read_text().splitlines():
                line = line.strip()
                if line.startswith("## "):
                    sec = line[3:]
                elif line and not line.startswith("#"):
                    out[line] = sec
            return out
    return out


def kind_of(text):
    zh, en = re.search(r"[\u4e00-\u9fff]", text), re.search(r"[A-Za-z]", text)
    return "混说" if zh and en else "英文" if en else "中文"


def pick(texts, kind, n, sec=(3.0, 9.0)):
    """按原文找录音；没录的，用同类（纯中文 / 纯英文 / 混说）、长度合适、以句号结尾的读稿句补上。"""
    t = takes()
    sec_of = script_sections()  # 「朗读」优先挑录音稿 11–15 段（念给人听的口吻）里的句子
    ids = [line_id(x) for x in texts if line_id(x) in t][:n]
    want = "中文" if kind == "朗读" else kind
    spare = sorted((lid for lid, e in t.items() if lid not in ids and not e.get("free")
                    and kind_of(e["text"]) == want and sec[0] <= e["sec"] <= sec[1]
                    and e["text"].rstrip().endswith(("。", ".", "！", "!"))),
                   key=lambda lid: (kind == "朗读" and "朗读" not in sec_of.get(t[lid]["text"].strip(), ""),
                                    abs(t[lid]["sec"] - 5)))
    ids += spare[: n - len(ids)]
    if len(ids) < n:
        raise SystemExit(f"录音不够：「{kind}」类的读稿句至少要 {n} 句，先去 /record 多录几句")
    return ids


# 试听句：都不在录音稿里，考它没听过的话
TESTS = [
    "嘿，我今天把声音克隆做出来了，你听听看像不像我？",
    "这个 model 是在本地跑的，完全不用上网，privacy 很好。",
    "Honestly, 我觉得它还差一点，but it's a good start.",
    "I think the best way to learn is to keep trying. 不要怕失败。",
    "不管发生什么事，记得手上永远都要有选择。",
]


def load_f5(ckpt=None):
    """F5-TTS 从本地 models/ 读（模型来自 ModelScope，Hugging Face 在国内连不上），只加载一次。"""
    import f5_tts_mlx.cfm as cfm
    import mlx.core as mx
    mx.set_cache_limit(1 << 30)  # 16G 的机器：显存缓存最多留 1G
    import f5_tts_mlx.generate as g
    from vocos_mlx import Vocos

    cfm.fetch_from_hub = lambda *a, **k: HERE / "models" / "f5-tts-v1"
    vocos_load = Vocos.from_pretrained.__func__
    Vocos.from_pretrained = classmethod(lambda cls, _: vocos_load(cls, str(HERE / "models" / "vocos-mel-24khz")))
    model = cfm.F5TTS.from_pretrained("local")
    if ckpt:  # clone_train.py 微调出来的权重
        import mlx.core as mx
        model.load_weights(list(mx.load(str(ckpt)).items()), strict=False)
    g.F5TTS = type("Cached", (), {"from_pretrained": staticmethod(lambda *a, **k: model)})
    return g.generate


def takes():
    t = {}
    for line in (REC / "takes.jsonl").read_text().splitlines():
        e = json.loads(line)
        if e.get("deleted"):
            t.pop(e["line_id"], None)
        elif "file" in e:
            t[e["line_id"]] = e
    return t


def load_trim(path):
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(1)
    n = int(len(a) * SR / sr)
    a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)
    # 20ms 一帧，能量低于最大值 -35dB 的头尾去掉，各留 0.1 秒
    f = SR // 50
    e = np.array([np.sqrt(np.mean(a[i:i + f] ** 2) + 1e-12) for i in range(0, len(a) - f, f)])
    on = np.where(20 * np.log10(e / e.max()) > -35)[0]
    s = max(0, on[0] * f - SR // 10)
    t = min(len(a), (on[-1] + 1) * f + SR // 10)
    return a[s:t]


def build_ref(name, out_dir):
    t = takes()
    parts, texts = [], []
    for lid in pick(REFS[name], name, 2):
        e = t[lid]
        parts += [load_trim(REC / e["file"]), np.zeros(int(SR * 0.3), np.float32)]
        texts.append(e["text"])
    a = np.concatenate(parts[:-1])
    a = a / (np.sqrt(np.mean(a ** 2)) + 1e-9) * 0.1  # 统一音量（F5 内部也按 0.1 RMS）
    path = out_dir / f"ref_{name}.wav"
    sf.write(path, a, SR)
    return path, " ".join(texts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", action="append", help="要念的话，可给多次")
    ap.add_argument("--ref", choices=list(REFS), action="append", help="参考组，默认全部")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--ckpt", default="", help="clone_train.py 微调出来的权重")
    args = ap.parse_args()

    generate = load_f5(args.ckpt or None)

    out = OUT / (datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + (f"-{Path(args.ckpt).stem}" if args.ckpt else ""))
    out.mkdir(parents=True)
    tests = args.text or TESTS
    rows = []
    for name in args.ref or list(REFS):
        ref_path, ref_text = build_ref(name, out)
        print(f"\n== 参考「{name}」：{ref_text}")
        for i, text in enumerate(tests):
            wav = out / f"{name}_{i}.wav"
            t0 = datetime.datetime.now()
            generate(generation_text=text, ref_audio_path=str(ref_path), ref_audio_text=ref_text,
                     steps=args.steps, estimate_duration=True, seed=7, output_path=str(wav))
            sec = (datetime.datetime.now() - t0).total_seconds()
            rows.append((name, ref_path.name, text, wav.name, sec))
            print(f"  {sec:.1f}s  {text}")
    write_page(out, rows)
    print(f"\n试听页：{out / 'index.html'}")


def write_page(out, rows):
    esc = html.escape
    by = {}
    for name, ref, text, wav, sec in rows:
        by.setdefault((name, ref), []).append((text, wav, sec))
    body = []
    for (name, ref), items in by.items():
        body.append(f"<h2>参考：{esc(name)}</h2><p class=ref>真人原声（参考）<audio controls src='{ref}'></audio></p>")
        for text, wav, sec in items:
            body.append(f"<div class=row><div>{esc(text)}<small>生成 {sec:.0f} 秒</small></div>"
                        f"<audio controls src='{wav}'></audio></div>")
    (out / "index.html").write_text(f"""<!doctype html><meta charset=utf-8><title>声音克隆试听</title>
<style>body{{font:16px -apple-system,sans-serif;max-width:820px;margin:24px auto;padding:0 16px;background:#fafafa;color:#222}}
h1{{font-size:22px}}h2{{font-size:17px;margin-top:28px;border-top:1px solid #ddd;padding-top:16px}}
.ref{{color:#666}}.row{{display:flex;gap:12px;align-items:center;justify-content:space-between;padding:8px 0;border-bottom:1px solid #eee}}
small{{display:block;color:#999;font-size:12px}}audio{{height:32px;flex-shrink:0}}</style>
<h1>声音克隆试听（零样本，F5-TTS）</h1>
<p>每组先听真人原声，再听克隆念的新句子。试听句都不在录音稿里。<b>以下声音由 AI 生成。</b></p>
{''.join(body)}""")


if __name__ == "__main__":
    main()
