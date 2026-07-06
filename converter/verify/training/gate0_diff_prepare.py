"""gate0_diff step 1 — stage identical inputs for both sides (C-layer).

Copies the S38 sovits gate's OUR-side 44k slices + their finished companions
(.soft.pt / .f0.npy / .spec.pt) into two FRESH dirs:

    TESTING/utai-v2-testing/diff_orig/gate/   (original side)
    TESTING/utai-v2-testing/diff_ours/gate/   (our side)

Feeding the same 44k wavs to both sides pins the comparison to the CODE axis
(the aug draw bounds depend on max|wav| — an input-axis difference would make
the augmentation products incomparable). Pre-placing soft/f0/spec makes
process_one skip those branches (skip-if-exists) so the original side needs
neither fairseq nor a GPU. .vol.npy is deliberately NOT copied — both sides
recompute it so the gate covers the vol path too.

Run (our venv):
    training\\.venv\\Scripts\\python.exe converter\\verify\\training\\gate0_diff_prepare.py
"""
import shutil
from pathlib import Path

SRC = Path(r"D:\MyDev\TESTING\utai-v2-testing\sovits_ours\dataset_44k\gate")
DSTS = [
    Path(r"D:\MyDev\TESTING\utai-v2-testing\diff_orig\gate"),
    Path(r"D:\MyDev\TESTING\utai-v2-testing\diff_ours\gate"),
]

COMPANION_SUFFIXES = (".wav", ".wav.soft.pt", ".wav.f0.npy", ".spec.pt")


def main():
    wavs = sorted(p for p in SRC.iterdir() if p.name.endswith(".wav"))
    assert wavs, f"no wavs in {SRC} — re-run the S38 sovits gate0 first"
    for dst in DSTS:
        if dst.parent.exists():
            shutil.rmtree(dst.parent)
        dst.mkdir(parents=True)
        n = 0
        for wav in wavs:
            stem = wav.name[: -len(".wav")]
            for suffix in COMPANION_SUFFIXES:
                src = SRC / (stem + suffix)
                assert src.exists(), f"missing companion {src}"
                shutil.copyfile(src, dst / src.name)
                n += 1
        print(f"staged {len(wavs)} slices ({n} files) -> {dst}")


if __name__ == "__main__":
    main()
