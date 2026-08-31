# -*- coding: utf-8 -*-
"""S169 gate — AMD arch-keyed device pick (hipenum.py + device.apply_amd_arch_mask
+ envtest.check_amd_device_pick).

判据房规(同 gate_resume_state.py):每条断言各问各的,变异要能各红各的;每个功能块
先有一条阳性对照(正确输入必须生效),再谈负例;「量不了」必须 FAIL,不许空绿。
本 gate 全程不需要 GPU、不 import torch(被测函数的 torch 只活在子进程/monkeypatch 后面),
在任何机器上都能跑。真机行为(780M 上 kernel 真跑、坏机上原生崩)由 S169 的实机冒烟盖,
这里钉的是选择逻辑与两条失败路径的【可归因性】:
  TRAINING_AMD_ENUM_FAILED   = 探针跑不起来
  TRAINING_AMD_GPU_NOT_FOUND = 探针跑了,没有匹配架构的设备
  ENVTEST_AMD_NO_COVERED_GPU = envtest 侧:没有该包驱动得动的设备
三个红必须互相分得开(S129 铁律:闸自己跑不起来 ≠ 被测的东西不对)。

env 卫生:每个动 os.environ 的块用 snapshot/restore 包住,收尾断言恢复成功
(S137 血训:工装自己写的状态要自证还原)。
"""
import os
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\MyDev\Utai_v2-dev\training")

from utai_train import device as device_shim  # noqa: E402
from utai_train import envtest  # noqa: E402
from utai_train import hipenum  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print("%-4s %s%s" % ("PASS" if cond else "FAIL", name, (" — " + detail) if (detail and not cond) else ""))


REPORTER_SHAPE = [
    {"index": 0, "name": "AMD Radeon(TM) Graphics", "arch": "gfx1035"},
    {"index": 1, "name": "AMD Radeon RX 7700S", "arch": "gfx1102"},
]

# ---------------- A. pick_hip_index (pure) ----------------
# A1 阳性对照:the exact field shape the fix exists for.
check("A1_reporter_shape_picks_the_dgpu", hipenum.pick_hip_index(REPORTER_SHAPE, "gfx1102") == 1)
# A2 absent arch → None (the caller raises NOT_FOUND — pick itself never invents).
check("A2_absent_arch_returns_none", hipenum.pick_hip_index(REPORTER_SHAPE, "gfx1103") is None)
# A3 gcnArchName feature suffixes must not defeat the match (Linux-style "gfx1102:sramecc-").
suffixed = [{"index": 0, "name": "X", "arch": "gfx1102:sramecc-:xnack-"}]
check("A3_arch_suffix_matches_base_token", hipenum.pick_hip_index(suffixed, "gfx1102") == 0)
# A4/A5 same-arch multi-GPU: name tiebreak wins, else first (deterministic).
twins = [
    {"index": 0, "name": "Radeon RX 7600", "arch": "gfx1102"},
    {"index": 1, "name": "AMD Radeon RX 7700S", "arch": "gfx1102"},
]
check("A4_prefer_name_breaks_ties", hipenum.pick_hip_index(twins, "gfx1102", prefer_name="AMD Radeon RX 7700S") == 1)
check("A5_no_name_match_takes_first", hipenum.pick_hip_index(twins, "gfx1102", prefer_name="nope") == 0)

# ---------------- B. enumerate_hip_devices (injected _run) ----------------


class FakeProc:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


def run_ok(cmd, **kw):
    return FakeProc(stdout="noise\n" + hipenum.ENUM_MARKER + '{"devices": [{"index": 1, "name": "RX", "arch": "gfx1102"}]}\n')


def run_garbage(cmd, **kw):
    return FakeProc(stdout="Traceback ...\n", stderr="ImportError: DLL load failed", returncode=1)


def run_badjson(cmd, **kw):
    return FakeProc(stdout=hipenum.ENUM_MARKER + "{not json\n")


def run_boom(cmd, **kw):
    raise subprocess.TimeoutExpired(cmd="x", timeout=1)


# B1 阳性对照:a valid child answer parses.
devs = hipenum.enumerate_hip_devices(_run=run_ok)
check("B1_valid_child_parses", devs == [{"index": 1, "name": "RX", "arch": "gfx1102"}])


def code_of(fn):
    try:
        fn()
        return None
    except RuntimeError as e:
        return str(e)


# B2/B3/B4 every broken-probe shape carries the ENUM_FAILED code — and ONLY that code.
for name, fn in [
    ("B2_no_marker_is_enum_failed", lambda: hipenum.enumerate_hip_devices(_run=run_garbage)),
    ("B3_child_exception_is_enum_failed", lambda: hipenum.enumerate_hip_devices(_run=run_boom)),
    ("B4_bad_payload_is_enum_failed", lambda: hipenum.enumerate_hip_devices(_run=run_badjson)),
]:
    msg = code_of(fn)
    check(name, msg is not None and msg.startswith(hipenum.ENUM_FAILED_CODE + ":"), repr(msg))

# B5 the child must see RAW ordinals: all three visibility vars stripped from its env
# even when the parent carries them (e.g. the S169 stopgap HIP_VISIBLE_DEVICES=1).
_snap = dict(os.environ)
seen_env = {}


def run_capture(cmd, **kw):
    seen_env.update(kw.get("env") or {})
    return run_ok(cmd, **kw)


try:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["HIP_VISIBLE_DEVICES"] = "1"
    os.environ["ROCR_VISIBLE_DEVICES"] = "2"
    hipenum.enumerate_hip_devices(_run=run_capture)
    check(
        "B5_child_env_strips_all_visibility_vars",
        seen_env and all(k not in seen_env for k in hipenum.VISIBILITY_VARS),
        str([k for k in hipenum.VISIBILITY_VARS if k in seen_env]),
    )
finally:
    os.environ.clear()
    os.environ.update(_snap)
check("B6_env_restored_after_B", dict(os.environ) == _snap)
# B7 the two failure CODEs are distinct literals (S129: two different reds).
check("B7_enum_and_notfound_codes_differ", hipenum.ENUM_FAILED_CODE != device_shim.AMD_GPU_NOT_FOUND_CODE)

# ---------------- C. device.apply_amd_arch_mask (monkeypatched enum) ----------------


def with_fake_enum(devices, fn):
    real = hipenum.enumerate_hip_devices
    hipenum.enumerate_hip_devices = lambda *a, **k: devices
    try:
        return fn()
    finally:
        hipenum.enumerate_hip_devices = real


# C1 the no-trigger pin: without gpu_gfx_target the env must be BYTE-IDENTICAL —
# this is the NV/CPU/pre-0.12.2 lane, and gate1's bitwise contract sits on it.
_snap = dict(os.environ)
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-uuid-sentinel"
    os.environ["HIP_VISIBLE_DEVICES"] = "7"
    before = dict(os.environ)
    device_shim.apply_amd_arch_mask({"device_backend": "cuda", "gpu": "0"})
    check("C1_no_trigger_key_is_a_byte_identical_noop", dict(os.environ) == before)

    # C2 阳性对照:the reporter shape, DXGI mask "0" → HIP ordinal "1"; stale user
    # overrides deleted, never blanked (Windows: empty == unset).
    with_fake_enum(
        REPORTER_SHAPE,
        lambda: device_shim.apply_amd_arch_mask(
            {"device_backend": "cuda", "gpu": "0", "gpu_gfx_target": "gfx1102"}
        ),
    )
    check("C2_mask_rekeyed_to_hip_ordinal", os.environ.get("CUDA_VISIBLE_DEVICES") == "1")
    check(
        "C2b_stale_overrides_deleted_not_blanked",
        "HIP_VISIBLE_DEVICES" not in os.environ and "ROCR_VISIBLE_DEVICES" not in os.environ,
    )

    # C3 no matching arch → the NOT_FOUND code, and NOT the enum code.
    msg = None
    try:
        with_fake_enum(
            REPORTER_SHAPE,
            lambda: device_shim.apply_amd_arch_mask(
                {"device_backend": "cuda", "gpu": "0", "gpu_gfx_target": "gfx1103"}
            ),
        )
    except RuntimeError as e:
        msg = str(e)
    check(
        "C3_no_match_is_gpu_not_found",
        msg is not None
        and msg.startswith(device_shim.AMD_GPU_NOT_FOUND_CODE + ":")
        and not msg.startswith(hipenum.ENUM_FAILED_CODE),
        repr(msg),
    )

    # C4 explicit CPU / xpu keep their semantics even with a stray target.
    before = dict(os.environ)
    device_shim.apply_amd_arch_mask({"device_backend": "cuda", "gpu": "-1", "gpu_gfx_target": "gfx1102"})
    device_shim.apply_amd_arch_mask({"device_backend": "xpu", "gpu": "0", "gpu_gfx_target": "gfx1102"})
    check("C4_cpu_and_xpu_lanes_untouched", dict(os.environ) == before)

    # C5 the auto lane (gpu "") still gets the mask — unmasked HIP defaults to device 0,
    # the exact wrong-silicon shape this function exists to prevent.
    with_fake_enum(
        REPORTER_SHAPE,
        lambda: device_shim.apply_amd_arch_mask(
            {"device_backend": "cuda", "gpu": "", "gpu_gfx_target": "gfx1102"}
        ),
    )
    check("C5_auto_lane_gets_the_mask_too", os.environ.get("CUDA_VISIBLE_DEVICES") == "1")

    # C6 name tiebreak flows through cfg.gpu_label.
    with_fake_enum(
        twins,
        lambda: device_shim.apply_amd_arch_mask(
            {
                "device_backend": "cuda",
                "gpu": "0",
                "gpu_gfx_target": "gfx1102",
                "gpu_label": "AMD Radeon RX 7700S",
            }
        ),
    )
    check("C6_gpu_label_breaks_ties", os.environ.get("CUDA_VISIBLE_DEVICES") == "1")
finally:
    os.environ.clear()
    os.environ.update(_snap)
check("C7_env_restored_after_C", dict(os.environ) == _snap)

# ---------------- D. envtest.check_amd_device_pick ----------------
_snap = dict(os.environ)
try:
    # D1/D2 skips: non-cuda tier, and no targets (NVIDIA packs / old callers).
    check("D1_cpu_tier_skips", envtest.check_amd_device_pick({"device": "cpu", "gfx_targets": ["gfx1102"]}) is None)
    check("D2_no_targets_skips", envtest.check_amd_device_pick({"device": "cuda", "gfx_targets": []}) is None)

    # D3 阳性对照:reporter shape + v2 target list → picks the dgpu, masks, says so.
    detail = with_fake_enum(
        REPORTER_SHAPE,
        lambda: envtest.check_amd_device_pick(
            {"device": "cuda", "gfx_targets": ["gfx1100", "gfx1101", "gfx1102", "gfx1103"]}
        ),
    )
    check(
        "D3_picks_and_masks_the_covered_gpu",
        os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
        and isinstance(detail, str)
        and "gfx1102" in detail
        and "RX 7700S" in detail,
        repr(detail),
    )

    # D4 nothing covered → its own CODE (distinct from both TRAINING_* reds).
    msg = None
    try:
        with_fake_enum(
            [{"index": 0, "name": "iGPU", "arch": "gfx1035"}],
            lambda: envtest.check_amd_device_pick({"device": "cuda", "gfx_targets": ["gfx1102"]}),
        )
    except RuntimeError as e:
        msg = str(e)
    check(
        "D4_uncovered_box_is_no_covered_gpu",
        msg is not None and msg.startswith("ENVTEST_AMD_NO_COVERED_GPU:"),
        repr(msg),
    )
finally:
    os.environ.clear()
    os.environ.update(_snap)
check("D5_env_restored_after_D", dict(os.environ) == _snap)

# D6 placement pin: after cuda_driver, before torch_backend (the first torch.cuda touch) —
# read from the CHECKS table itself, not the source text.
names = [n for n, _ in envtest.CHECKS]
check(
    "D6_check_order_precedes_hip_init",
    names.index("cuda_driver") < names.index("amd_device_pick") < names.index("torch_backend"),
    str(names[:5]),
)
# D7 short-circuit pin: a pick FAIL must break the run like cuda_driver's does. The break
# lives in main()'s loop; pin the condition tuple (needle split so this file can't satisfy
# it) on COMMENT-STRIPPED source — a commented-out condition must not count — and require
# the `break` action within the same block (S169 review: condition-only pins survive the
# action being deleted).
env_src = open(os.path.join(r"D:\MyDev\Utai_v2-dev\training", "utai_train", "envtest.py"), encoding="utf-8").read()
env_nc = "\n".join(ln.split("#")[0] for ln in env_src.splitlines())
_i = env_nc.find('name in ("cuda_driver", "amd_' + 'device_pick") and status == "fail"')
check(
    "D7_pick_fail_short_circuits",
    _i >= 0 and "break" in env_nc[_i:_i + 400],
    "condition@%d" % _i,
)

print("\n%d PASS / %d FAIL" % (len(PASS), len(FAIL)))
if FAIL:
    print("FAILED:", ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
