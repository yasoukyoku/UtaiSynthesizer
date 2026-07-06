"""gate1_diff step 1 — stage the training-equivalence run.

Both sides train on the SAME preprocessed products (gate0_diff's OUR-side dir,
which now holds wav/soft/f0/spec/vol/mel/aug_mel/aug_vol), from the SAME
vec768 base model, with IDENTICAL yamls (fp32 CPU, num_workers 0,
cache_all_data, batch 4, interval_log 1, interval_val 8, force_save 16,
epochs 3 -> ceil(31/4)*3 = 24 steps, 3 validation boundaries — the FIRST one
carries the lazy NSF-HiFiGAN Generator construction's RNG block, the later
ones don't; both must align step-for-step). infer.speedup 50 keeps the CPU
test() inference bearable (identical on both sides — sampling is RNG-free
except the initial randn, which both consume identically).

Creates:
    TESTING/gate1_diff_filelists/{train,val}.txt   (absolute, forward-slash)
    TESTING/gate1_diff_orig/   (expdir, model_0.pt seeded)  + yaml
    TESTING/gate1_diff_ours/   (expdir, model_0.pt seeded)
    TESTING/gate1_diff_ours_ws/diffusion.yaml       (our driver's workspace)

Run (our venv):
    training\\.venv\\Scripts\\python.exe converter\\verify\\training\\gate1_diff_prepare.py
NB re-running WIPES both expdirs (like the S38 sovits gate1 prepare).
"""
import os
import random
import shutil

import yaml

UTAI = r"D:\MyDev\Utai_v2-dev"
SOVITS = r"D:\MyDev\so-vits-svc\so-vits-svc"
TESTING = r"D:\MyDev\TESTING\utai-v2-testing"
DATA = os.path.join(TESTING, "diff_ours", "gate")  # gate0 our-side products
BASE = os.path.join(UTAI, "data", "models", "training", "sovits", "diffusion", "vec768", "model_0.pt")
NSF = os.path.join(UTAI, "data", "models", "training", "sovits", "nsf_hifigan", "model")
SEED = 1234


def build_filelists(flist_dir):
    """Same split recipe as our flist.build_filelists (seeded, val = first 2)."""
    wavs = sorted(
        os.path.join(DATA, n).replace("\\", "/")
        for n in os.listdir(DATA)
        if n.endswith(".wav")
    )
    rng = random.Random(SEED)
    rng.shuffle(wavs)
    train, val = wavs[2:], wavs[:2]
    rng.shuffle(train)
    rng.shuffle(val)
    os.makedirs(flist_dir, exist_ok=True)
    for name, rows in (("train.txt", train), ("val.txt", val)):
        with open(os.path.join(flist_dir, name), "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
    return len(train), len(val)


def make_yaml(expdir, flist_dir):
    with open(
        os.path.join(SOVITS, "configs_template", "diffusion_template.yaml"),
        encoding="utf-8",
    ) as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["encoder"] = "vec768l12"
    cfg["data"]["encoder_out_channels"] = 768
    cfg["data"]["training_files"] = os.path.join(flist_dir, "train.txt").replace("\\", "/")
    cfg["data"]["validation_files"] = os.path.join(flist_dir, "val.txt").replace("\\", "/")
    cfg["model"]["n_spk"] = 1
    cfg["model"]["k_step_max"] = 0
    cfg["spk"] = {"gate": 0}
    cfg["device"] = "cpu"
    cfg["vocoder"]["ckpt"] = NSF.replace("\\", "/")
    cfg["env"]["expdir"] = expdir.replace("\\", "/")
    cfg["env"]["gpu_id"] = 0
    cfg["train"]["num_workers"] = 0
    cfg["train"]["amp_dtype"] = "fp32"
    cfg["train"]["batch_size"] = 4
    cfg["train"]["epochs"] = 3
    cfg["train"]["interval_log"] = 1
    cfg["train"]["interval_val"] = 8
    cfg["train"]["interval_force_save"] = 16
    cfg["infer"]["speedup"] = 50
    return cfg


def main():
    assert os.path.isfile(os.path.join(DATA, "000_000.wav.mel.npy")), (
        "gate0_diff our-side products missing — run gate0_diff_run_ours.py first"
    )
    flist_dir = os.path.join(TESTING, "gate1_diff_filelists")
    n_train, n_val = build_filelists(flist_dir)
    print(f"filelists: {n_train} train / {n_val} val")

    for side in ("gate1_diff_orig", "gate1_diff_ours"):
        expdir = os.path.join(TESTING, side)
        if os.path.isdir(expdir):
            shutil.rmtree(expdir)
        os.makedirs(expdir)
        shutil.copyfile(BASE, os.path.join(expdir, "model_0.pt"))
        print(f"seeded base -> {expdir}")

    with open(
        os.path.join(TESTING, "gate1_diff_orig.yaml"), "w", encoding="utf-8"
    ) as f:
        yaml.dump(make_yaml(os.path.join(TESTING, "gate1_diff_orig"), flist_dir), f)

    ws = os.path.join(TESTING, "gate1_diff_ours_ws")
    os.makedirs(ws, exist_ok=True)
    with open(os.path.join(ws, "diffusion.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(make_yaml(os.path.join(TESTING, "gate1_diff_ours"), flist_dir), f)

    print("GATE1 DIFF PREPARE DONE")


if __name__ == "__main__":
    main()
