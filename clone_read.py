"""用克隆的声音念长篇文字（书、信），全部在本地跑。

用法（.venv-clone 环境）：
  .venv-clone/bin/python clone_read.py examples/yuqishu.txt --ref 中文 --ckpt data/clone_ft/step_900.safetensors
  .venv-clone/bin/python clone_read.py examples/odyssey_book1.txt --ref 英文 --max-words 3000

把文字按句子切成 20 秒以内的小段，逐段生成（参考音频、随机种子固定，前后声音一致），
段落之间多停一下，最后拼成一个 wav。每段生成完就存，断了重跑会从断点接着念。
输出在 data/longform/out/<文件名>-<参考>-<模型>/，带一个试听页。
"""
import argparse
import html
import json
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf

import clone

SR = clone.SR


def chunks(text, max_zh=45, max_en=40):
    """按段落、句子切；中文按字数，英文按词数，一段不超过上限。"""
    out = []  # (text, 段尾?)
    for para in [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]:
        para = re.sub(r"\s*\n\s*", " ", para)
        if para.isupper():  # 全大写的标题模型会念成乱码，转成首字母大写
            para = para.title().replace("’S", "’s")
        sents = re.findall(r".+?(?:[。！？!?；;]+[”’\"]?|[.]+[”’\"]?\s|$)", para)
        sents = [s.strip() for s in sents if s.strip()]
        # 太长的句子再按逗号切
        lim = lambda s: max_en if is_en(s) else max_zh
        sents = [x for s in sents for x in (re.findall(r".+?(?:[，、：,:]|$)", s) if size(s) > lim(s) else [s]) if x.strip()]
        cur = ""
        for s in sents:
            if cur and size(cur + " " + s) > (max_en if is_en(s) else max_zh):
                out.append([cur, False])
                cur = s
            else:
                cur = (cur + " " + s).strip() if is_en(s) else cur + s
        if cur:
            out.append([cur, False])
        out[-1][1] = True
    return out


def is_en(s):
    return len(re.findall(r"[A-Za-z]", s)) > len(re.findall(r"[一-鿿]", s))


def size(s):
    return len(s.split()) if is_en(s) else len(re.findall(r"[一-鿿]", s))


def speaking_rate():
    """从读稿录音量出你平时的语速：纯中文句 → 每字几秒，纯英文句 → 每字母几秒（去掉头尾静音，取中位数）。"""
    zh, en = [], []
    for e in clone.takes().values():
        t = e["text"]
        if e.get("free"):
            continue
        n_zh, n_en = len(re.findall(r"[一-鿿]", t)), len(re.findall(r"[A-Za-z]", t))
        if n_zh >= 8 and n_en == 0:
            zh.append(len(clone.load_trim(clone.REC / e["file"])) / SR / n_zh)
        elif n_en >= 20 and n_zh == 0:
            en.append(len(clone.load_trim(clone.REC / e["file"])) / SR / n_en)
    return float(np.median(zh)), float(np.median(en))


ASR = "mlx-community/whisper-large-v3-turbo"


def tokens(s):
    """比对用：汉字转无声调拼音（同音字不算错），英文转小写单词。"""
    from pypinyin import lazy_pinyin
    out = []
    for w in re.findall(r"[\u4e00-\u9fff]+|[A-Za-z']+", s):
        out += lazy_pinyin(w) if re.match(r"[\u4e00-\u9fff]", w) else [w.lower()]
    return out


def score(text, wav):
    """用 Whisper 听一遍，和原文按拼音/单词比对，返回 0–1 和听到的文字。"""
    import difflib
    import mlx_whisper
    r = mlx_whisper.transcribe(str(wav), path_or_hf_repo=ASR, language="en" if is_en(text) else "zh",
                               condition_on_previous_text=False)
    a, b = tokens(text), tokens(r["text"])
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    hit = sum(m.size for m in sm.get_matching_blocks())
    # 漏念和多念都扣分；另外原文里没有、念出来却有一串 4 个字/词原样出现两次的，算重复，直接判不过
    sc = 2 * hit / max(1, len(a) + len(b))
    if repeats(b) - repeats(a):
        sc = min(sc, 0.8)
    return sc, r["text"].strip()


def repeats(x, n=4):
    seen, rep = set(), set()
    for i in range(len(x) - n + 1):
        g = tuple(x[i:i + n])
        (rep if g in seen else seen).add(g)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--ref", choices=list(clone.REFS), default="中文")
    ap.add_argument("--engine", choices=["f5", "sovits"], default="f5",
                    help="f5 = F5-TTS（不可商用，.venv-clone）；sovits = GPT-SoVITS（MIT，.venv-sovits）")
    ap.add_argument("--ckpt", default="", help="F5 微调后的权重；不给就用原版（零样本）")
    ap.add_argument("--gpt", default="", help="GPT-SoVITS 微调出来的 GPT 权重（.ckpt）")
    ap.add_argument("--sovits", default="", help="GPT-SoVITS 微调出来的 SoVITS 权重（.pth）；auto = 训练轮数最多的那个")
    ap.add_argument("--aux", type=int, default=0, help="GPT-SoVITS 额外参考段数（音色融合，6 段最好）")
    ap.add_argument("--max-words", type=int, default=0, help="英文只念前 N 个词（约 150 词/分钟）")
    ap.add_argument("--speed", type=float, default=0.9, help="相对你平时说话的语速；念书默认慢一成")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--tries", type=int, default=3, help="质检不过就换随机种子重念，最多几次")
    ap.add_argument("--min-score", type=float, default=0.85)
    ap.add_argument("--recheck", action="store_true", help="已经念好的段也重新质检，不过的重念")
    ap.add_argument("--assemble", action="store_true", help="不念，只把已经念好的段拼起来（念到一半想先听）")
    args = ap.parse_args()
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    src = Path(args.file).resolve()  # GPT-SoVITS 加载时会切换工作目录
    text = src.read_text()
    if args.max_words:
        words, n, keep = text.split("\n"), 0, []
        for line in words:
            keep.append(line)
            n += len(line.split())
            if n >= args.max_words and line.rstrip().endswith((".", "?", "!", "”")):
                break
        text = "\n".join(keep)
    parts = chunks(text)

    if args.engine == "sovits" and args.sovits == "auto":
        import sovits as _sv
        _ws = sorted((_sv.ENGINE / "SoVITS_weights_v2ProPlus").glob("me_e*.pth"),
                     key=lambda p: int(re.search(r"_e(\d+)", p.stem).group(1)))
        args.sovits = str(_ws[-1]) if _ws else "auto"  # auto = 训练轮数最多的那个
        print(f"用 {Path(args.sovits).name}")
    if args.engine == "sovits":
        tag = "sovits-" + (Path(args.sovits).stem if args.sovits else "zeroshot") + (f"-gpt{Path(args.gpt).stem}" if args.gpt else "") + (f"-aux{args.aux}" if args.aux else "")
    else:
        tag = Path(args.ckpt).stem if args.ckpt else "zeroshot"
    out = clone.HERE / "data" / "longform" / "out" / f"{src.stem}-{args.ref}-{tag}"  # 输出只放 data/（不进 git）
    seg = out / "seg"
    seg.mkdir(parents=True, exist_ok=True)
    ref_path, ref_text = clone.build_ref(args.ref, out)

    if args.assemble:
        qc_path = out / "qc.json"
        assemble(out, src, parts, json.loads(qc_path.read_text()) if qc_path.exists() else {}, args, tag)
        return
    t0 = time.time()
    qc_path = out / "qc.json"
    qc = json.loads(qc_path.read_text()) if qc_path.exists() else {}
    if args.engine == "sovits":
        synth = sovits_synth(args, out)
    else:
        synth = f5_synth(args, ref_path, ref_text)
    print(f"{len(parts)} 段，引擎 {args.engine}，参考「{args.ref}」，模型 {tag}")
    run_parts(args, parts, out, seg, qc, qc_path, synth, t0, src, tag)


def f5_synth(args, ref_path, ref_text):
    clone.load_f5(args.ckpt or None)
    import f5_tts_mlx.generate as g
    import mlx.core as mx
    from f5_tts_mlx.utils import convert_char_to_pinyin
    model = g.F5TTS.from_pretrained()
    ref, _ = sf.read(ref_path, dtype="float32")
    ref_mx = mx.array(ref)[None]
    zh_sec, en_sec = speaking_rate()
    print(f"语速 中文 {args.speed / zh_sec:.1f} 字/秒、英文 {args.speed / en_sec:.1f} 字母/秒")

    def synth(t, seed):
        # 时长 = 字数 × 你平时念一个字/字母的时间（从读稿录音量出来）。
        # 给多了模型会把多出来的时间拿去重复句尾，所以不能按参考那两句估
        zh, letters = len(re.findall(r"[\u4e00-\u9fff]", t)), len(re.sub(r"[^A-Za-z]", "", t))
        gen_sec = (zh * zh_sec + letters * en_sec) / args.speed
        dur = int((len(ref) / SR + gen_sec) * SR / 256)
        w, _ = model.sample(ref_mx, text=convert_char_to_pinyin([ref_text + " " + t]), duration=dur,
                            steps=args.steps, method="rk4", cfg_strength=2.0,
                            sway_sampling_coef=-1.0, seed=seed)
        return np.array(w[len(ref):])
    return synth


def sovits_synth(args, out):
    """GPT-SoVITS：时长由它自己的 GPT 决定，不用估。输出 32k，转成 24k 和 F5 那边统一。"""
    import sovits
    if args.sovits == "auto":
        raise SystemExit("还没有微调好的 SoVITS 权重，先跑 sovits_train.py")
    tts = sovits.load(args.gpt or None, args.sovits or None)
    ref_path, ref_text = sovits.ref_for(args.ref)
    sf.write(out / f"ref_{args.ref}.wav", sf.read(ref_path)[0], SR)

    def synth(t, seed):
        a = sovits.say(tts, t, ref_path, ref_text, seed=seed, speed=args.speed, aux=sovits.aux_refs(args.aux))
        n = int(len(a) * SR / sovits.SR)
        return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)
    return synth


def run_parts(args, parts, out, seg, qc, qc_path, synth, t0, src, tag):
    for i, (t, _) in enumerate(parts):
        wav = seg / f"{i:04d}.wav"
        key = str(i)
        if wav.exists() and qc.get(key, {}).get("text") == t and not (args.recheck and qc[key]["score"] < args.min_score):
            continue
        if wav.exists() and args.recheck and key not in qc:
            sc, heard = score(t, wav)
            qc[key] = dict(text=t, score=sc, heard=heard, seed=7)
            if sc >= args.min_score:
                qc_path.write_text(json.dumps(qc, ensure_ascii=False, indent=1))
                continue
        best = qc.get(key) if wav.exists() else None
        for k in range(args.tries):
            seed = 7 + k + (100 if best else 0)
            w = synth(t, seed)
            tmp = seg / f"{i:04d}.try.wav"
            sf.write(tmp, w, SR)
            sc, heard = score(t, tmp)
            if not best or sc > best["score"]:
                tmp.replace(wav)
                best = dict(text=t, score=sc, heard=heard, seed=seed)
            else:
                tmp.unlink()
            if best["score"] >= args.min_score:
                break
        qc[key] = best
        qc_path.write_text(json.dumps(qc, ensure_ascii=False, indent=1))
        el = time.time() - t0
        flag = "" if best["score"] >= args.min_score else "  ⚠ 质检没过"
        if (i + 1) % 5 == 0:
            assemble(out, src, parts, qc, args, tag)
        print(f"[{i + 1}/{len(parts)}] 质检 {best['score']:.0%}（试 {k + 1} 次）累计 {el / 60:.1f} 分钟  {t[:30]}{flag}", flush=True)

    assemble(out, src, parts, qc, args, tag)


def assemble(out, src, parts, qc, args, tag):
    """把念好的段拼成 full.wav 和试听页。还没念完时只拼前面连续念好的部分，页面上标进度。"""
    seg = out / "seg"
    audio, n = [], 0
    for i, (t, end) in enumerate(parts):
        f = seg / f"{i:04d}.wav"
        if not f.exists():
            break
        audio += [sf.read(f, dtype="float32")[0], np.zeros(int(SR * (0.7 if end else 0.25)), np.float32)]
        n += 1
    if not audio:
        return
    full = np.concatenate(audio)
    full = full / max(1e-6, np.abs(full).max()) * 0.9
    sf.write(out / "full.wav", full, SR)
    (out / "parts.json").write_text(json.dumps(parts, ensure_ascii=False, indent=1))
    mins = len(full) / SR / 60
    doing = "" if n == len(parts) else f" · <b>还在念：{n}/{len(parts)} 段，刷新可听到更多</b>"
    paras = []
    for i, (t, _) in enumerate(parts):
        note = ""
        if i >= n:
            note = " <small>（还没念）</small>"
        elif qc.get(str(i), {}).get("score", 1) < args.min_score:
            note = " <small>⚠ 这段可能念得不准</small>"
        paras.append(f"<p{' class=todo' if i >= n else ''}>{html.escape(t)}{note}</p>")
    (out / "index.html").write_text(f"""<!doctype html><meta charset=utf-8><title>{html.escape(src.stem)}</title>
<style>body{{font:17px/1.8 -apple-system,sans-serif;max-width:760px;margin:24px auto;padding:0 16px;color:#222}}
audio{{width:100%;position:sticky;top:0}}p{{margin:0 0 12px}}small{{color:#888}}.todo{{color:#bbb}}</style>
<h1>{html.escape(src.stem)}</h1><small>由 AI 用克隆声音朗读 · 参考「{args.ref}」· 模型 {tag} · {mins:.1f} 分钟{doing}</small>
<audio controls src="full.wav?n={n}"></audio>
{''.join(paras)}""")
    if n == len(parts):
        bad = [i for i in range(len(parts)) if qc.get(str(i), {}).get("score", 1) < args.min_score]
        print(f"质检：{len(parts) - len(bad)}/{len(parts)} 段通过" + (f"，没过的段：{bad}" if bad else ""))
        print(f"完成：{mins:.1f} 分钟 → {out / 'full.wav'}")


if __name__ == "__main__":
    main()
