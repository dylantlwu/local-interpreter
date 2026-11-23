"""微调 F5-TTS，让克隆的声音更像你。数据 = 读稿录音（文字已知）+ 校对过的自由讲。

用法（.venv-clone 环境）：
  .venv-clone/bin/python clone_train.py --steps 20            # 先试几步，看内存和速度
  .venv-clone/bin/python clone_train.py --steps 1200 --save-every 300

每 save-every 步存一次 data/clone_ft/step_N.safetensors，并算一次验证损失
（留出约 8% 的句子不训练，随机数固定，所以不同步数之间能直接比）。
之后用 clone.py --ckpt data/clone_ft/step_N.safetensors 试听。
"""
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.optimizers import AdamW, cosine_decay, join_schedules, linear_schedule, clip_grad_norm
from mlx.utils import tree_flatten

import clone
from f5_tts_mlx.utils import convert_char_to_pinyin

OUT = clone.HERE / "data" / "clone_ft"


def items():
    """读稿的每一句 + 已校对的自由讲（过长的跳过）。"""
    out = []
    for lid, e in clone.takes().items():
        text = e.get("transcript") if e.get("free") else e["text"]
        if e.get("free") and not e.get("checked"):
            continue
        if not text or e["sec"] > 25:
            continue
        out.append((lid, text, clone.REC / e["file"]))
    return out


def heldout(lid):
    return int(hashlib.sha1(lid.encode()).hexdigest(), 16) % 12 == 0


def prepare(model, rows):
    data = []
    for lid, text, path in rows:
        a = clone.load_trim(path)
        a = a / (np.sqrt(np.mean(a ** 2)) + 1e-9) * 0.1
        mel = model._mel_spec(mx.array(a)[None])
        if mel.shape[-1] != 100:
            mel = mx.transpose(mel, (0, 2, 1))  # → b n d
        data.append(dict(id=lid, text=text, mel=mel, chars=convert_char_to_pinyin([text])))
    mx.eval([d["mel"] for d in data])
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=300)
    ap.add_argument("--freeze", type=int, default=6, help="冻结前 N 层 transformer（省内存）")
    ap.add_argument("--max-sec", type=float, default=10.0, help="去掉静音后超过这个秒数的句子不训练（省内存）")
    args = ap.parse_args()

    mx.set_cache_limit(1 << 30)  # 16G 的机器：显存缓存最多留 1G，不然会被系统杀掉
    clone.load_f5()
    import f5_tts_mlx.generate as g
    model = g.F5TTS.from_pretrained()

    rows = items()
    train = prepare(model, [r for r in rows if not heldout(r[0])])
    valid = prepare(model, [r for r in rows if heldout(r[0])])
    cap = int(args.max_sec * 24000 / 256)
    long_ = [d for d in train if d["mel"].shape[1] > cap]
    train = [d for d in train if d["mel"].shape[1] <= cap]
    print(f"跳过 {len(long_)} 句超过 {args.max_sec:.0f} 秒的")
    sec = sum(d["mel"].shape[1] for d in train) * 256 / 24000
    print(f"训练 {len(train)} 句（{sec / 60:.1f} 分钟），验证 {len(valid)} 句")

    model.freeze(keys=["_mel_spec"], strict=False)
    model._mel_spec.freeze()
    for blk in model.transformer.transformer_blocks[: args.freeze]:
        blk.freeze()
    n = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"可训练参数 {n / 1e6:.0f}M")

    sched = join_schedules([linear_schedule(1e-8, args.lr, args.warmup),
                            cosine_decay(args.lr, max(1, args.steps - args.warmup), args.lr * 0.1)],
                           [args.warmup])
    opt = AdamW(learning_rate=sched, weight_decay=1e-2)

    def loss_fn(m, mel, chars):
        return m(mel, text=chars)

    grad_fn = nn.value_and_grad(model, loss_fn)

    def val_loss():
        model.eval()
        mx.random.seed(1234)
        ls = [model(d["mel"], text=d["chars"]).item() for d in valid for _ in range(4)]
        model.train()
        return float(np.mean(ls))

    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / "train.log", "a")

    def note(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    note(f"== {time.strftime('%F %T')} steps={args.steps} lr={args.lr} freeze={args.freeze}")
    note(f"step 0 验证损失 {val_loss():.4f}")
    model.train()
    rng = random.Random(0)
    order = []
    t0 = time.time()
    run = 0.0
    for step in range(1, args.steps + 1):
        if not order:
            order = list(range(len(train)))
            rng.shuffle(order)
        d = train[order.pop()]
        mel, chars = d["mel"], d["chars"]
        loss, grads = grad_fn(model, mel, chars)
        grads, _ = clip_grad_norm(grads, 1.0)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        run = loss.item() if step == 1 else run * 0.95 + loss.item() * 0.05
        if step % 10 == 0 or step == args.steps:
            el = time.time() - t0
            note(f"step {step} 训练损失 {run:.4f}  {el / step:.2f}s/步  "
                 f"峰值内存 {mx.get_peak_memory() / 1e9:.1f}GB")
        if step % args.save_every == 0 or step == args.steps:
            path = OUT / f"step_{step}.safetensors"
            # mlx 的 save_safetensors 遇到中文路径会崩（段错误），所以切到目录里用相对文件名存
            cwd = os.getcwd()
            os.chdir(OUT)
            try:
                mx.save_safetensors(path.name, dict(tree_flatten(model.parameters())))
            finally:
                os.chdir(cwd)
            note(f"step {step} 验证损失 {val_loss():.4f}  存到 {path.name}")


if __name__ == "__main__":
    main()
