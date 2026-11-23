"""纠错数据工具：用会后纠错页面（/review）改好的句子做自动评测和 LoRA 训练集。

  .venv312/bin/python corrections.py stats                 看攒了多少句
  .venv312/bin/python corrections.py eval                  用正在运行的服务把纠错过的句子重译一遍，和正确答案比打分
  .venv312/bin/python corrections.py eval --heldout        只评留出来的那 1/10（做过 LoRA 之后要看这个，别拿训练过的句子考自己）
  .venv312/bin/python corrections.py export                导出 LoRA 训练集到 data/lora/（train.jsonl / valid.jsonl）
  .venv312/bin/python corrections.py voice                 录音攒了多少分钟（「保存我的录音」开着时存的）
  .venv312/bin/python corrections.py voice --export        导出校对过的录音 + 文字到 data/voice_dataset/（声音克隆用）
  .venv312/bin/python corrections.py import-log server.log 把加这个功能之前的服务日志里的翻译导进纠错页面

评分用 chrF（按字符比较，0–100，越高越接近正确答案；中英文都适用），另外看完全一致的比例。
换模型比较：用 MT_MODEL=… 重启服务，再跑一次 eval，结果都存在 data/evals/ 里。
"""
import argparse
import hashlib
import json
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"


def read_jsonl(path):
    out = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def gold_items():
    """纠错过的句子：原文/译文取改过的版本；跳过的不要。"""
    fixes = {c["id"]: c for c in read_jsonl(DATA / "corrections.jsonl")}
    out = []
    for f in sorted((DATA / "history").glob("*.jsonl")) if (DATA / "history").is_dir() else []:
        for it in read_jsonl(f):
            fx = fixes.get(it["id"])
            if fx and fx["status"] in ("ok", "fixed"):
                out.append({**it, "status": fx["status"], "gold_src": fx.get("src") or it["src"],
                            "gold_tr": fx.get("tr") or it["tr"], "note": fx.get("note", "")})
    return out, fixes


def heldout(it):
    """按 id 固定分出 1/10 不拿去训练，专门用来考。"""
    return int(hashlib.sha1(it["id"].encode()).hexdigest(), 16) % 10 == 0


def _ngrams(s, n):
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def chrf(hyp, ref, max_n=6, beta=2.0):
    hyp, ref = re.sub(r"\s+", "", hyp), re.sub(r"\s+", "", ref)
    if not hyp or not ref:
        return 100.0 if hyp == ref else 0.0
    ps, rs = [], []
    for n in range(1, max_n + 1):
        h, r = _ngrams(hyp, n), _ngrams(ref, n)
        if not h or not r:
            continue
        hit = sum((h & r).values())
        ps.append(hit / sum(h.values()))
        rs.append(hit / sum(r.values()))
    p, r = sum(ps) / len(ps), sum(rs) / len(rs)
    return 0.0 if p + r == 0 else 100 * (1 + beta ** 2) * p * r / (beta ** 2 * p + r)


def norm(s):
    return re.sub(r"[\s，。！？、,.!?'\"“”‘’]+", "", s).lower()


def translate(server, text, direction, history):
    body = json.dumps({"text": text, "direction": direction, "history": history, "no_record": True}).encode()
    req = urllib.request.Request(f"{server}/translate", data=body, headers={"Content-Type": "application/json"})
    out, first = "", None
    with urllib.request.urlopen(req, timeout=120) as r:
        for line in r:
            line = line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            msg = json.loads(line[6:])
            if "t" in msg:
                out += msg["t"]
            if msg.get("error"):
                raise RuntimeError(msg["error"])
            if msg.get("done"):
                first = msg.get("first_ms")
    return out.strip(), first


def cmd_stats(a):
    hist = [it for f in sorted((DATA / "history").glob("*.jsonl")) for it in read_jsonl(f)] if (DATA / "history").is_dir() else []
    golds, fixes = gold_items()
    st = Counter(f["status"] for f in fixes.values())
    print(f"记下的句子 {len(hist)}，看过 {len(fixes)}（对 {st['ok']}，改过 {st['fixed']}，跳过 {st['skip']}），待看 {len(hist) - len(fixes)}")
    by_dir = Counter(g["dir"] for g in golds)
    print(f"可用来评测/训练 {len(golds)} 句：中→英 {by_dir['zh2en']}，英→中 {by_dir['en2zh']}；其中留出考试用 {sum(map(heldout, golds))} 句")
    asr = sum(1 for g in golds if g["gold_src"] != g["src"])
    print(f"识别（原文）改过 {asr} 句，翻译改过 {sum(1 for g in golds if g['gold_tr'] != g['tr'])} 句")
    t = teacher_items()
    print(f"老师案例（teacher/cases.tsv）{len(t)} 句：" + "，".join(f"{k} {v}" for k, v in Counter(x["cat"] for x in t).items()))
    if len(golds) < 200:
        print("（LoRA 一般要几百句改过的才有明显效果；先攒着，评测现在就能用）")


def cmd_eval(a):
    golds, _ = gold_items()
    if a.heldout:
        golds = [g for g in golds if heldout(g)]
    if a.dir:
        golds = [g for g in golds if g["dir"] == a.dir]
    if not golds:
        sys.exit("还没有纠错过的句子。先打开 http://localhost:8765/review 改几句。")
    try:
        info = json.load(urllib.request.urlopen(f"{a.server}/info", timeout=5))
    except OSError:
        sys.exit(f"连不上服务 {a.server}，先启动 server.py")
    model = info["mt_model"].rsplit("/", 1)[-1]
    print(f"用 {model} 重译 {len(golds)} 句……")
    rows = []
    for i, g in enumerate(golds, 1):
        # 用纠错过的原文翻：评的是翻译，不让识别错误混进来
        hyp, first = translate(a.server, g["gold_src"], g["dir"], g.get("history") or [])
        rows.append({"id": g["id"], "dir": g["dir"], "src": g["gold_src"], "gold": g["gold_tr"], "hyp": hyp, "old": g["tr"],
                     "status": g["status"], "chrf": chrf(hyp, g["gold_tr"]), "old_chrf": chrf(g["tr"], g["gold_tr"]),
                     "exact": norm(hyp) == norm(g["gold_tr"]), "first_ms": first})
        print(f"\r  {i}/{len(golds)}", end="", flush=True)
    print()
    n = len(rows)
    avg = sum(r["chrf"] for r in rows) / n
    exact = sum(r["exact"] for r in rows) / n
    fixed = [r for r in rows if r["status"] == "fixed"]
    oks = [r for r in rows if r["status"] == "ok"]
    firsts = sorted(r["first_ms"] for r in rows if r["first_ms"] is not None)
    print(f"\n== {model}  {n} 句{'（只看留出的）' if a.heldout else ''}    首字中位数 {firsts[len(firsts) // 2] if firsts else '-'}ms")
    # 主分数只看"当时译错、改过"的句子：它们的正确答案是人（或老师）写的。
    # "当时就译对"的句子，正确答案就是旧模型的原话，换个说法也扣分，所以只报改动了几句，好坏要人看。
    if fixed:
        print(f"当时译错的 {len(fixed)} 句：chrF {sum(r['chrf'] for r in fixed) / len(fixed):.1f}"
              f"（当时 {sum(r['old_chrf'] for r in fixed) / len(fixed):.1f}），完全译对 {sum(r['exact'] for r in fixed)} 句，"
              f"比当时更接近 {sum(r['chrf'] > r['old_chrf'] + 1 for r in fixed)} 句")
    if oks:
        changed = [r for r in oks if not r["exact"]]
        print(f"当时就译对的 {len(oks)} 句：{len(oks) - len(changed)} 句一字不差，{len(changed)} 句换了说法（不一定错，列出来给人看）")
        for r in changed[:a.show]:
            print(f"    {r['src']}\n      当时：{r['gold']}\n      现在：{r['hyp']}")
    print(f"全部 {n} 句 chrF {avg:.1f}，完全一致 {exact:.0%}（受换说法影响，只适合看大方向）")
    print("\n当时译错、现在离正确答案还最远的几句：")
    for r in sorted(fixed or rows, key=lambda r: r["chrf"])[:a.show]:
        print(f"  [{r['chrf']:.0f}] {r['src']}\n       正确：{r['gold']}\n       现在：{r['hyp']}")
    out = DATA / "evals" / f"{time.strftime('%Y%m%d-%H%M%S')}-{model}{'-heldout' if a.heldout else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": info["mt_model"], "scene": info.get("scene"), "n": n, "chrf": avg, "exact": exact, "rows": rows},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n明细存在 {out.relative_to(HERE)}")


def teacher_items():
    """teacher/cases.tsv：大模型老师写的教学案例（方向、类别、原文、正确译文、可选的上一句原文和译文）。"""
    out = []
    path = HERE / "teacher" / "cases.tsv"
    if not path.exists():
        return out
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split("\t")
        d, cat, src, tr = f[:4]
        hist = []
        if len(f) >= 6:  # 上一句：方向按有没有汉字判断
            who = "me" if re.search(r"[\u4e00-\u9fff]", f[4]) else "them"
            hist = [{"who": who, "text": f[4], "tr": f[5]}]
        out.append({"id": f"t{n}", "dir": d, "cat": cat, "gold_src": src, "gold_tr": tr, "history": hist})
    return out


def cmd_export(a):
    sys.argv = sys.argv[:1]
    import server  # 只借用提示词，不加载模型

    golds, _ = gold_items()
    exam = {norm(g["gold_src"]) for g in golds}
    teach = [t for t in teacher_items() if norm(t["gold_src"]) not in exam]  # 和考试原句一样的不拿去练
    split = {"train": [], "valid": []}
    for t in teach:  # 老师案例：9/10 训练，1/10 看训练有没有过头
        split["valid" if heldout(t) else "train"].append(t)
    if a.include_real:  # 真实会议句子也拿去练（留出的 1/10 仍然不练）
        split["train"] += [g for g in golds if not heldout(g)]
    if not split["train"]:
        sys.exit("没有可训练的句子。")
    if not split["valid"]:  # mlx_lm 要求有 valid.jsonl
        split["valid"].append(split["train"].pop())
    out = HERE / a.out
    out.mkdir(parents=True, exist_ok=True)
    for k, rows in split.items():
        with open(out / f"{k}.jsonl", "w", encoding="utf-8") as f:
            for g in rows:
                msgs = server.final_turns(g["gold_src"], g["dir"], g.get("history") or []) + [{"role": "assistant", "content": g["gold_tr"]}]
                f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
    print(f"导出到 {out.relative_to(HERE)}/：训练 {len(split['train'])} 句，验证 {len(split['valid'])} 句"
          f"（老师案例 {len(teach)}{'，含真实会议句子' if a.include_real else '；真实会议句子只用来考试'}）")
    print("训练（先停掉服务，省内存）：")
    print(f"  .venv312/bin/python -m mlx_lm lora --model {server.MT_MODEL} --train --data {a.out} "
          f"--iters 300 --batch-size 1 --num-layers 8 --mask-prompt --adapter-path data/adapters")


def voice_items():
    """存下来的录音，配上纠错后的文字。status：fixed/ok = 文字已校对；todo = 还没看；skip = 不要（杂音或不是本人）。"""
    vdir = DATA / "voice"
    clips = {c["id"]: c for f in sorted(vdir.glob("*.jsonl")) for c in read_jsonl(f)} if vdir.is_dir() else {}
    fixes = {c["id"]: c for c in read_jsonl(DATA / "corrections.jsonl")}
    hist = {it["voice_id"]: it for f in sorted((DATA / "history").glob("*.jsonl")) for it in read_jsonl(f) if it.get("voice_id")} \
        if (DATA / "history").is_dir() else {}
    out = []
    for vid, c in clips.items():
        it = hist.get(vid)
        fx = fixes.get(it["id"]) if it else None
        status = fx["status"] if fx else "todo"
        text = (fx.get("src") if fx else "") or c["text"]
        out.append({**c, "text": text, "asr_text": c["text"], "status": status, "path": vdir / c["file"]})
    # 读稿录音（/record）：读稿的文字本来就是对的；自由讲要在页面上改过文字才算校对
    takes = {}
    for t in read_jsonl(DATA / "recording" / "takes.jsonl"):
        if t.get("deleted"):
            takes.pop(t["line_id"], None)
        elif "file" in t:
            takes[t["line_id"]] = t
        elif t["line_id"] in takes:
            takes[t["line_id"]] = {**takes[t["line_id"]], **t}
    for t in takes.values():
        text = t.get("transcript", "") if t["free"] else t["text"]
        lang = "en" if not re.search(r"[\u4e00-\u9fff]", text) else "zh"
        status = ("ok" if t.get("checked") else "todo") if t["free"] else "ok"
        out.append({"id": t["file"], "file": t["file"], "text": text, "asr_text": t.get("asr_text", text), "lang": lang,
                    "sec": t["sec"], "speech_sec": t["speech_sec"], "status": status, "path": DATA / "recording" / t["file"],
                    "source": "free" if t["free"] else "script"})
    return out


def cmd_voice(a):
    clips = voice_items()
    if not clips:
        sys.exit("还没有录音。在同传页面勾上「保存我的录音」再说话。")
    mins = lambda xs: sum(x["speech_sec"] for x in xs) / 60  # noqa: E731
    by = {k: [c for c in clips if c["status"] == k] for k in ("ok", "fixed", "todo", "skip")}
    good = by["ok"] + by["fixed"]
    src = Counter(c.get("source", "") for c in clips)
    print(f"录音 {len(clips)} 段，人声共 {mins(clips):.1f} 分钟（读稿 {src['script']}，自由讲 {src['free']}，会议/现场 {src['me'] + src['live']}）")
    print(f"  已校对 {len(good)} 段 {mins(good):.1f} 分钟（其中识别有错、改过 {sum(c['text'] != c['asr_text'] for c in good)} 段）")
    print(f"  待校对 {len(by['todo'])} 段 {mins(by['todo']):.1f} 分钟，跳过 {len(by['skip'])} 段")
    m = mins(good)
    goal = "可以试零样本克隆（像个大概）" if m < 10 else "够做声音微调了（会很像）" if m >= 30 else "可以做声音微调（基本像）"
    print(f"  {goal}；一般 0.5 分钟能克隆个大概，10 分钟以上可以微调，30 分钟以上会很像")
    if a.export:
        out = HERE / a.export
        (out / "wavs").mkdir(parents=True, exist_ok=True)
        import shutil

        rows = []
        for c in good:
            name = c["file"].replace("/", "_")
            shutil.copy(c["path"], out / "wavs" / name)
            rows.append(f"wavs/{name}|{c['text']}|{c['lang']}")
        (out / "metadata.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(f"导出 {len(rows)} 段到 {out.relative_to(HERE)}/（wavs/ + metadata.csv：文件|文字|语种，常见的语音训练格式）")


def cmd_import_log(a):
    """旧日志的格式：HH:MM:SS 翻译 zh2en 首字218ms 共994ms: 原文 => 译文"""
    path = Path(a.log)
    date = time.strftime("%Y-%m-%d", time.localtime(path.stat().st_mtime))
    pat = re.compile(r"^(\d\d:\d\d:\d\d) 翻译 (zh2en|en2zh) 首字(\d+|None)ms 共\d+ms: (.*?) => (.*)$")
    history, rows = [], []
    known = {it["id"] for f in (DATA / "history").glob("*.jsonl") for it in read_jsonl(f)} if (DATA / "history").is_dir() else set()
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        m = pat.match(line)
        if not m:
            continue
        t, d, first, src, tr = m.groups()
        ts = time.mktime(time.strptime(f"{date} {t}", "%Y-%m-%d %H:%M:%S"))
        rid = f"{int(ts * 1000) + i}"
        if rid not in known:
            rows.append({"id": rid, "time": f"{date} {t}", "dir": d, "src": src, "tr": tr,
                         "first_ms": None if first == "None" else int(first), "scene": "", "model": "（导入的旧日志）",
                         "history": history[-4:]})
        history.append({"who": "me" if d == "zh2en" else "them", "text": src, "tr": tr})
    if rows:
        out = DATA / "history" / f"{date}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"导入 {len(rows)} 句（日期按文件修改时间算：{date}）")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats")
    e = sub.add_parser("eval")
    e.add_argument("--server", default="http://localhost:8765")
    e.add_argument("--heldout", action="store_true", help="只评留出的 1/10")
    e.add_argument("--dir", choices=["zh2en", "en2zh"])
    e.add_argument("--show", type=int, default=8, help="列出差距最大的几句")
    x = sub.add_parser("export")
    x.add_argument("--out", default="data/lora")
    x.add_argument("--include-real", action="store_true", help="真实会议里纠错过的句子也拿去训练（默认只用来考试）")
    v = sub.add_parser("voice", help="录音统计；--export 导出校对过的录音和文字")
    v.add_argument("--export", nargs="?", const="data/voice_dataset", help="导出目录（默认 data/voice_dataset）")
    i = sub.add_parser("import-log")
    i.add_argument("log")
    a = p.parse_args()
    {"stats": cmd_stats, "eval": cmd_eval, "export": cmd_export, "voice": cmd_voice, "import-log": cmd_import_log}[a.cmd](a)
