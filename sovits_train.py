"""用你的录音微调 GPT-SoVITS（v2ProPlus），全部在本地 CPU 上跑。

用法（.venv-sovits 环境；先跑 .venv-clone/bin/python sovits_prep.py 生成清单）：
  .venv-sovits/bin/python sovits_train.py                 # 预处理 + SoVITS 16 轮（GPT 默认不微调）
  .venv-sovits/bin/python sovits_train.py --skip-prep

做的事和官方 WebUI 的「一键三连」+「训练」一样，只是不开网页：
  1a 文字 → 音素 + BERT 特征    1b 声音 → HuBERT 特征 + 说话人向量    1c 声音 → 语音 token
  SoVITS（声音像不像）和 GPT（语气、停顿）分开训练，权重存到 engines/GPT-SoVITS/*_weights_v2ProPlus/
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).parent
ENGINE = HERE / "engines" / "GPT-SoVITS"
VERSION = "v2ProPlus"
EXP = "me"
PM = "GPT_SoVITS/pretrained_models"


def run(script, env):
    e = dict(os.environ, **{k: str(v) for k, v in env.items()},
             NLTK_DATA=str(HERE / "engines" / "nltk" / "nltk_data"), PYTHONPATH=f"{ENGINE}:{ENGINE}/GPT_SoVITS")
    print(f"→ {script}", flush=True)
    subprocess.run([sys.executable, "-s", script], cwd=ENGINE, env=e, check=True)


def prep(lst):
    opt = f"logs/{EXP}"
    base = dict(inp_text=lst, inp_wav_dir="", exp_name=EXP, opt_dir=opt, i_part=0, all_parts=1,
                is_half=False, _CUDA_VISIBLE_DEVICES=0)
    run("GPT_SoVITS/prepare_datasets/1-get-text.py", dict(base, bert_pretrained_dir=f"{PM}/chinese-roberta-wwm-ext-large"))
    os.replace(ENGINE / opt / "2-name2text-0.txt", ENGINE / opt / "2-name2text.txt")
    run("GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py", dict(base, cnhubert_base_dir=f"{PM}/chinese-hubert-base",
                                                                 sv_path=f"{PM}/sv/pretrained_eres2netv2w24s4ep4.ckpt"))
    run("GPT_SoVITS/prepare_datasets/2-get-sv.py", dict(base, sv_path=f"{PM}/sv/pretrained_eres2netv2w24s4ep4.ckpt"))
    run("GPT_SoVITS/prepare_datasets/3-get-semantic.py",
        dict(base, pretrained_s2G=f"{PM}/v2Pro/s2Gv2ProPlus.pth", s2config_path=f"GPT_SoVITS/configs/s2{VERSION}.json"))
    sem = ENGINE / opt / "6-name2semantic-0.tsv"
    (ENGINE / opt / "6-name2semantic.tsv").write_text("item_name\tsemantic_audio\n" + sem.read_text())
    sem.unlink()


def train_sovits(epochs, batch):
    data = json.loads((ENGINE / f"GPT_SoVITS/configs/s2{VERSION}.json").read_text())
    t = data["train"]
    t.update(fp16_run=False, batch_size=batch, epochs=epochs, text_low_lr_rate=0.4,
             pretrained_s2G=f"{PM}/v2Pro/s2Gv2ProPlus.pth", pretrained_s2D=f"{PM}/v2Pro/s2Dv2ProPlus.pth",
             if_save_latest=True, if_save_every_weights=True, save_every_epoch=4, gpu_numbers="0",
             grad_ckpt=False, lora_rank=32)
    data["model"]["version"] = data["version"] = VERSION
    data["data"]["exp_dir"] = data["s2_ckpt_dir"] = f"logs/{EXP}"
    data["save_weight_dir"] = f"SoVITS_weights_{VERSION}"
    data["name"] = EXP
    os.makedirs(ENGINE / f"logs/{EXP}/logs_s2_{VERSION}", exist_ok=True)
    os.makedirs(ENGINE / "TEMP", exist_ok=True)
    os.makedirs(ENGINE / data["save_weight_dir"], exist_ok=True)  # 不建的话训练完存权重会报错
    (ENGINE / "TEMP/tmp_s2.json").write_text(json.dumps(data))
    e = dict(os.environ, PYTHONPATH=f"{ENGINE}:{ENGINE}/GPT_SoVITS")
    subprocess.run([sys.executable, "-s", "GPT_SoVITS/s2_train.py", "--config", "TEMP/tmp_s2.json"], cwd=ENGINE, env=e, check=True)


def train_gpt(epochs, batch):
    data = yaml.safe_load((ENGINE / "GPT_SoVITS/configs/s1longer-v2.yaml").read_text())
    data["train"].update(precision="32", batch_size=batch, epochs=epochs, save_every_n_epoch=5,
                         if_save_every_weights=True, if_save_latest=True, if_dpo=False,
                         half_weights_save_dir=f"GPT_weights_{VERSION}", exp_name=EXP)
    data["pretrained_s1"] = f"{PM}/s1v3.ckpt"
    data["train_semantic_path"] = f"logs/{EXP}/6-name2semantic.tsv"
    data["train_phoneme_path"] = f"logs/{EXP}/2-name2text.txt"
    data["output_dir"] = f"logs/{EXP}/logs_s1_{VERSION}"
    os.makedirs(ENGINE / data["output_dir"], exist_ok=True)
    os.makedirs(ENGINE / f"GPT_weights_{VERSION}", exist_ok=True)
    (ENGINE / "TEMP/tmp_s1.yaml").write_text(yaml.dump(data, default_flow_style=False))
    e = dict(os.environ, PYTHONPATH=f"{ENGINE}:{ENGINE}/GPT_SoVITS", hz="25hz", _CUDA_VISIBLE_DEVICES="0")
    subprocess.run([sys.executable, "-s", "GPT_SoVITS/s1_train.py", "--config_file", "TEMP/tmp_s1.yaml"], cwd=ENGINE, env=e, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-prep", action="store_true")
    # 实测 16 轮声纹最像（8→24 轮里 16 轮最好）；微调 GPT 反而更容易漏字，默认不训
    ap.add_argument("--sovits-epochs", type=int, default=16)
    ap.add_argument("--gpt-epochs", type=int, default=0, help="0 = 不微调 GPT，用原版")
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()
    if not args.skip_prep:
        prep(str(HERE / "data" / "sovits" / "train.list"))
    train_sovits(args.sovits_epochs, args.batch)
    if args.gpt_epochs:
        train_gpt(args.gpt_epochs, args.batch)
    print("== 训练完成")
    for d in (f"SoVITS_weights_{VERSION}", f"GPT_weights_{VERSION}"):
        for f in sorted((ENGINE / d).glob(f"{EXP}*")):
            print("  ", f.relative_to(ENGINE))


if __name__ == "__main__":
    main()
