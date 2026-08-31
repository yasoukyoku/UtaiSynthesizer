"""HIP device enumeration + arch-keyed selection for the AMD lane (S169).

Why this exists: on Windows the HIP runtime's device ordinal space is NOT the
DXGI adapter order the app's GPU list is built from (S169 field case: HIP put an
unsupported gfx1035 iGPU at index 0 and the selected RX 7700S / gfx1102 at
index 1, so masking with the DXGI-derived "0" trained on the wrong silicon and
died with hipErrorInvalidImage). There is no cross-API LUID/UUID bridge on the
AMD lane, but every HIP device reports its gfx arch (gcnArchName) and the Rust
side knows the picked adapter's arch from its PCI id — so the arch IS the key.

Enumeration must happen in a CHILD process:
  - visibility env vars are read once at HIP-runtime init, so the parent cannot
    enumerate first and then re-mask itself;
  - on a machine with an uncovered arch the runtime can crash natively at the
    first kernel launch — a child turns that into a diagnosable error instead of
    taking the trainer down with it (property queries alone are kernel-free and
    were safe on the S169 field machine, but the isolation costs nothing).

Module contract: torch-free, stdlib-only at top level (device.setup_visibility's
"never import torch" seam extends to everything it can reach). Two distinct
failure CODEs, per the closed-gate iron rule ("the tool could not run" and "the
thing under test is wrong" must never share a red):
  - TRAINING_AMD_ENUM_FAILED    — the enum child could not produce a device list
  - TRAINING_AMD_GPU_NOT_FOUND  — raised by the CALLER (device.py) when the list
                                  contains no device with the wanted arch
"""
import json
import os
import subprocess
import sys

ENUM_MARKER = "HIPENUM_JSON "

# "The probe could not run" — kept a plain top-level literal: the Rust cross-language
# gate parses it as the single source (same contract as ckpt_guard.py's CODE).
ENUM_FAILED_CODE = "TRAINING_AMD_ENUM_FAILED"

# Every device-visibility var HIP/CUDA honours. The enum child strips ALL of them
# so its indices are RAW HIP ordinals; the trainer strips them too before setting
# its own mask, so a stale user-level override (e.g. the S169 stopgap
# HIP_VISIBLE_DEVICES=1) cannot recompose with ours into an ambiguous view.
VISIBILITY_VARS = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")

# The child prints one marker line of JSON and hard-exits: the Windows HIP runtime
# is known to crash/hang during interpreter teardown, and os._exit keeps a
# successful enumeration from being reported as a crash (same posture as
# runner.py / envtest.py). Property queries launch no kernels.
_CHILD_SRC = (
    "import json, os\n"
    "import torch\n"
    "devs = []\n"
    "if torch.cuda.is_available():\n"
    "    for i in range(torch.cuda.device_count()):\n"
    "        p = torch.cuda.get_device_properties(i)\n"
    "        devs.append({'index': i, 'name': p.name,\n"
    "                     'arch': getattr(p, 'gcnArchName', '')})\n"
    "print('" + ENUM_MARKER.strip() + " ' + json.dumps({'devices': devs}), flush=True)\n"
    "os._exit(0)\n"
)


def base_arch(arch):
    """gcnArchName may carry feature suffixes ("gfx1102:sramecc-:xnack-") —
    comparisons use the base token."""
    return str(arch).split(":")[0]


def enumerate_hip_devices(python_exe=None, timeout=900, _run=None):
    """Enumerate HIP devices in a clean child process.

    Returns a list of {"index": int, "name": str, "arch": str} in RAW HIP order.
    Raises RuntimeError("TRAINING_AMD_ENUM_FAILED: ...") when the child cannot
    run or does not produce a parseable list — that is the "tool could not run"
    red, distinct from the caller's "no matching device" red.

    `_run` injects a subprocess.run-compatible callable for tests; production
    always uses the real one with this pack's own interpreter.

    The 900 s ceiling mirrors pyenv.rs's ENVTEST_TIMEOUT rationale (S169 review):
    on the TRAINING lane this child performs the machine's very first torch-hip
    import, and an antivirus cold-scanning the multi-GB site-packages can stall
    that for minutes — a fleet condition the 15-minute envtest ceiling was
    explicitly calibrated for. A tighter cap would refuse starts on exactly the
    pathological-AV boxes that previously (slowly) trained.
    """
    exe = python_exe or sys.executable
    env = dict(os.environ)
    for k in VISIBILITY_VARS:
        env.pop(k, None)
    run = _run or subprocess.run
    try:
        proc = run(
            [exe, "-c", _CHILD_SRC],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception as e:
        raise RuntimeError("%s: enum child failed to run: %r" % (ENUM_FAILED_CODE, e))
    line = None
    for ln in (proc.stdout or "").splitlines():
        if ln.startswith(ENUM_MARKER):
            line = ln
    if line is None:
        tail = " / ".join((proc.stderr or "").strip().splitlines()[-5:])
        raise RuntimeError(
            "%s: enum child exit=%r gave no device list; stderr tail: %s"
            % (ENUM_FAILED_CODE, proc.returncode, tail or "<empty>")
        )
    try:
        devs = json.loads(line[len(ENUM_MARKER):])["devices"]
    except Exception as e:
        raise RuntimeError("%s: bad enum payload: %r" % (ENUM_FAILED_CODE, e))
    return devs


def pick_hip_index(devices, gfx_target, prefer_name=None):
    """Pure selection: the RAW HIP ordinal of the first device whose base arch
    equals `gfx_target`. Among several matches (multi-GPU rigs with identical
    cards) a device whose name equals `prefer_name` wins, else the first match —
    identical arch means identical kernels, so any match is correct to train on.
    Returns None when nothing matches (the caller raises TRAINING_AMD_GPU_NOT_FOUND
    with the visible-device inventory).

    ⚠ Known limit (S169, registered): TWO IDENTICAL cards (same arch AND same
    name) collapse to the first HIP match — picking "the second RX 7700S" in the
    UI may train the first one. There is no cross-API key to tell twins apart on
    the AMD lane (no LUID/UUID bridge), and guessing by list position would be
    right half the time at best. Kernels and results are identical either way;
    only the physical-card preference is lost."""
    want = base_arch(gfx_target)
    matches = [d for d in devices if base_arch(d.get("arch", "")) == want]
    if not matches:
        return None
    if prefer_name:
        named = [d for d in matches if d.get("name") == prefer_name]
        if named:
            matches = named
    return int(matches[0]["index"])


def describe_devices(devices):
    """One-line inventory for error messages and logs."""
    return ", ".join(
        "%d:%s(%s)" % (int(d.get("index", -1)), d.get("name", "?"), d.get("arch", "?"))
        for d in devices
    ) or "<no HIP devices>"
