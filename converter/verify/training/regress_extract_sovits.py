"""S39 regression check — extract.py is a SHARED file of the S38-live-tested
sovits main pipeline and gained the diff_mode branch; prove the NON-diff path
still produces byte-identical products.

Runs extract_all(diff_mode=False) on a FRESH copy of the gate 44k wavs (no
caches) and compares every product (.soft.pt / .f0.npy / .spec.pt / .vol.npy)
against the S38-era outputs archived in sovits_ours. All fp32 CPU — every
stage is deterministic, so anything but exact equality is a regression.
.npy compare byte-wise; .pt compare by EXACT tensor equality — torch.save
embeds the destination basename as the zip archive root, and the S38 baseline
predates the atomic-write (tmp+rename) fix, so its archive roots differ
(".soft" vs ".soft.pt"; measured S39: tensors exactly equal, max_abs 0.0).

Run (our venv):
    training\\.venv\\Scripts\\python.exe converter\\verify\\training\\regress_extract_sovits.py
"""
import filecmp
import os
import shutil
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "training"))

TESTING = r"D:\MyDev\TESTING\utai-v2-testing"
BASELINE = os.path.join(TESTING, "sovits_ours", "dataset_44k", "gate")
FRESH_ROOT = os.path.join(TESTING, "regress_extract")
FRESH = os.path.join(FRESH_ROOT, "gate")

PRODUCT_SUFFIXES = (".wav.soft.pt", ".wav.f0.npy", ".spec.pt", ".wav.vol.npy")


class _Stop:
    def requested(self):
        return False

    def check(self):
        pass


class _Reporter:
    def stage(self, stage, done=None, total=None, message=None):
        pass


def main():
    from utai_train.sovits import utils
    from utai_train.sovits.extract import extract_all

    if os.path.isdir(FRESH_ROOT):
        shutil.rmtree(FRESH_ROOT)
    os.makedirs(FRESH)
    wavs = sorted(n for n in os.listdir(BASELINE) if n.endswith(".wav"))
    for n in wavs:
        shutil.copyfile(os.path.join(BASELINE, n), os.path.join(FRESH, n))
    print(f"staged {len(wavs)} wavs (no caches)")

    hps = utils.get_hparams_from_file(
        os.path.join(TESTING, "sovits_ours", "config.json")
    )
    extract_all(
        FRESH_ROOT,
        hps,
        os.path.join(REPO, "data", "models", "auxiliary", "contentvec_768l12.onnx"),
        os.path.join(REPO, "data", "models", "training", "sovits", "rmvpe.pt"),
        "cpu",
        _Reporter(),
        _Stop(),
        # deliberately DEFAULT diff args — this is the S38 call shape
    )

    import torch

    def equal(a, b):
        if a.endswith(".pt"):
            ta = torch.load(a, map_location="cpu", weights_only=True)
            tb = torch.load(b, map_location="cpu", weights_only=True)
            return ta.shape == tb.shape and bool(torch.equal(ta, tb))
        return filecmp.cmp(a, b, shallow=False)

    bad = 0
    checked = 0
    for n in wavs:
        stem = n[: -len(".wav")]
        for suffix in PRODUCT_SUFFIXES:
            a = os.path.join(BASELINE, stem + suffix)
            b = os.path.join(FRESH, stem + suffix)
            if not os.path.exists(a):
                continue  # baseline lacks it (e.g. vol when vol_embedding off)
            checked += 1
            if not (os.path.exists(b) and equal(a, b)):
                bad += 1
                print(f"MISMATCH {stem + suffix}")
    # the fresh dir must NOT have grown diff products in non-diff mode
    stray = [
        n for n in os.listdir(FRESH)
        if n.endswith((".mel.npy", ".aug_mel.npy", ".aug_vol.npy"))
    ]
    print(f"byte-compared {checked} products, mismatches: {bad}, stray diff products: {len(stray)}")
    ok = bad == 0 and not stray
    print("REGRESS EXTRACT SOVITS:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
