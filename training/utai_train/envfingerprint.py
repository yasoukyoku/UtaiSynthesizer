"""One line block describing the environment this training run actually got.

WHY THIS EXISTS. The last three community bug reports each needed a follow-up question whose
answer the app already had: "where is your data dir" (S168), "which GPU index" (S169), "do you
have Visual Studio" (S172). Each round trip cost the reporter days. These facts are free at run
start, so a single log file should be enough to triage without writing back.

DESIGN RULES, learned from those same three:

  * Report what THIS PROCESS RECEIVED, never what someone intended to send. The S169 bug lived
    exactly in that gap -- the gate promised one thing and the code handed over another -- so a
    parent-side line about its own intent would have been worse than nothing.
  * Never let the fingerprint kill a run. Every probe is individually guarded; a fingerprint
    that raises would turn a diagnostic into an outage.
  * Say "unknown" out loud. A field we could not read is a different fact from a field that is
    empty, and collapsing them is how a log starts lying.
  * The C++ toolchain probe is a HINT and is labelled as one. It looks for the things clang's
    MSVC detection uses; it does not compile anything, so it must not be phrased as a verdict.
"""

import os
import sys


def _torch_facts():
    try:
        import torch
    except Exception as e:  # torch import failure is itself the most useful line we could print
        return ["torch=UNIMPORTABLE (%s: %s)" % (type(e).__name__, e)]
    out = ["torch=%s" % torch.__version__]
    out.append("cuda=%s" % (torch.version.cuda or "-"))
    out.append("hip=%s" % (getattr(torch.version, "hip", None) or "-"))
    try:
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            out.append("devices=%d" % n)
            for i in range(n):
                try:
                    props = torch.cuda.get_device_properties(i)
                    # gcnArchName exists on ROCm builds and is the fact that matters there;
                    # on CUDA builds it is absent and the compute capability takes its place.
                    arch = getattr(props, "gcnArchName", None) or "sm_%d%d" % (
                        props.major,
                        props.minor,
                    )
                    out.append("dev%d=%r arch=%s" % (i, props.name, arch))
                except Exception as e:
                    out.append("dev%d=UNREADABLE (%s)" % (i, type(e).__name__))
        else:
            out.append("devices=0 (torch.cuda.is_available()=False)")
    except Exception as e:
        out.append("devices=UNREADABLE (%s: %s)" % (type(e).__name__, e))
    return out


def _env_facts():
    # The compile options are the load-bearing one: this is the ONLY place that can prove the
    # S172 header injection actually reached the process that needed it. -idirafter on a
    # directory that does not exist is silently ignored, so without this line a machine that
    # never got the fix and a machine that got it and broke elsewhere produce identical logs.
    keys = [
        "HIPRTC_COMPILE_OPTIONS_APPEND",
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "MIOPEN_FIND_MODE",
        "MIOPEN_LOG_LEVEL",
        "MIOPEN_USER_DB_PATH",
        "PYTORCH_ENABLE_XPU_FALLBACK",
    ]
    return ["%s=%s" % (k, os.environ.get(k, "-")) for k in keys]


def _header_payload_facts():
    """Did the shipped C++ headers arrive, counted the way the app counts them?

    The cwd is <install>/training and the injected paths are relative to it, so this resolves
    exactly what the compiler will resolve -- not a path we hope is equivalent."""
    root = os.path.join(os.getcwd(), "rocm_cxx_headers")
    if not os.path.isdir(root):
        return ["cxx_headers=ABSENT at %s" % root]
    out = []
    for sub in ("v1", "clangres"):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            out.append("%s=ABSENT" % sub)
            continue
        try:
            n = sum(len(f) for _r, _d, f in os.walk(d))
            out.append("%s=%d files" % (sub, n))
        except OSError as e:
            out.append("%s=UNREADABLE (%s)" % (sub, type(e).__name__))
    return ["cxx_headers=%s" % " ".join(out)]


def _host_cxx_hint():
    """Whether this machine looks like it has its own C++ headers.

    A HINT, and labelled as one -- it probes what clang's MSVC toolchain detection probes, and
    compiles nothing. It is here because it is the single fact that separated the machines that
    worked in S172 from the ones that did not, and we had to ASK for it."""
    if os.name != "nt":
        return "host_cxx=n/a (not windows)"
    if os.environ.get("INCLUDE"):
        return "host_cxx=likely (INCLUDE is set)"
    for var in ("ProgramFiles", "ProgramFiles(x86)"):
        # `pf_root`, not `base`: converter/verify/training/gate_pool_table.py enumerates every
        # os.path.join(<name>, "<literal>") under training/ and holds that set of base names to
        # exact equality, so a vague name here reads to it like a new POOL directory. It is also
        # simply a better name for what this is.
        pf_root = os.environ.get(var)
        if pf_root and os.path.isdir(os.path.join(pf_root, "Microsoft Visual Studio")):
            return "host_cxx=likely (%s\\Microsoft Visual Studio exists)" % var
    return "host_cxx=none found (no INCLUDE, no Visual Studio directory)"


def emit(prefix="[env]"):
    """Print the fingerprint to stderr. Never raises: a diagnostic must not end a run."""
    try:
        parts = []
        parts.append("python=%s" % sys.version.split()[0])
        parts.append("cwd=%s" % os.getcwd())
        parts += _torch_facts()
        parts += _header_payload_facts()
        parts.append(_host_cxx_hint())
        parts += _env_facts()
        for p in parts:
            print("%s %s" % (prefix, p), file=sys.stderr)
        sys.stderr.flush()
    except Exception as e:  # noqa: BLE001 - deliberately total
        try:
            print("%s fingerprint failed: %s: %s" % (prefix, type(e).__name__, e),
                  file=sys.stderr, flush=True)
        except Exception:
            pass
