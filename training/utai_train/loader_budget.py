"""DataLoader in-flight memory budget (S114 §F5-1).

Community bug: a run "just froze". The log says

    RuntimeError: Couldn't open shared file mapping: <torch_6836_575016820_874>,
                  error code: <1455>

and 1455 is Windows ERROR_COMMITMENT_LIMIT -- "The paging file is too small for
this operation to complete". Every batch a DataLoader worker hands back travels
through a shared file mapping charged against the system COMMIT limit, and there
are ``num_workers * prefetch_factor`` of them alive at once. On the reporting
machine (16 GB RAM, ~20 GB commit, and a vocal render running in the same app --
ORT reported ``bad allocation`` in the same minute) that ran the limit dry. The
exception is raised inside multiprocessing's *daemon feeder thread*, which prints
and keeps going, so the main process simply waits on a queue that will never be
fed: no crash, no error, an infinite hang.

WHY ADAPTIVE AND NOT "JUST LOWER THE DEFAULTS"
----------------------------------------------
Upstream RVC already hit this and shipped a one-size-fits-all mitigation; the
comment is still sitting above the loader we vendored from them::

    # It is possible that dataloader's workers are out of shared memory. ...
    # num_workers=8 -> num_workers=4

They halved it for EVERYONE to save the small machines. Lowering our defaults
again would repeat that trade -- a 64 GB box would pay for a 16 GB box's problem,
and training throughput matters. So this module is REDUCE-ONLY and budget-driven:

  * budget fits  -> the caller's requested numbers are returned UNCHANGED, so a
    roomy machine behaves byte-for-byte like it does today;
  * budget short -> step down a documented ladder (prefetch first, then workers,
    finally 0 = in-process, which uses no file mappings at all);
  * either way the decision AND its inputs are logged, because a number nobody
    can see is a number nobody can check.

⚠ HONEST BOUNDARY -- read before trusting ``BUDGET_FRACTION``
-------------------------------------------------------------
``batch_bytes`` is MEASURED (one real batch, see ``probe_batch_bytes``).
``BUDGET_FRACTION`` is NOT. Validating it needs several machines with different
RAM running real datasets, and this project has exactly one dev box. It is a
conservative design choice, it is reduce-only, and it is logged on every run so
the first field report can settle it. Do not describe it as measured.

⚠ And the mirror of that: "a big machine could go FASTER with more workers" is a
PERFORMANCE claim, not a safety one, and worker count is not monotonic in speed.
Nothing here ever raises the caller's request. See the queue entry §F5-① for what
measuring that would take.

⚠ Upstream is a witness that the FAILURE is real. It is not a witness that
upstream's numbers are good -- those repos are "runs at all" quality, and we have
already found bugs in them that nobody upstream had noticed.
"""

import logging
import os
import random
import sys

_LOG = logging.getLogger(__name__)

#: Share of the CURRENTLY AVAILABLE commit the in-flight batches may claim.
#: Unvalidated (see the module doc). Conservative on purpose: the rest has to
#: cover the model, the CUDA context, pinned staging buffers, the app itself and
#: -- as the report showed -- possibly a vocal render in the same process tree.
BUDGET_FRACTION = 0.25

#: torch's own default; the ladder never goes below it while workers remain.
MIN_PREFETCH = 2


def available_commit_bytes():
    """Bytes that can still be charged against the system commit limit, or None.

    This is the exact quantity error 1455 runs out of, so it is the right thing to
    budget against -- NOT free physical RAM, which stays comfortable while commit
    is exhausted. Windows only (``GlobalMemoryStatusEx.ullAvailPageFile``); every
    other platform returns None and the caller then changes nothing.
    """
    if not sys.platform.startswith("win"):
        return None
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return None
        return int(st.ullAvailPageFile)
    except Exception:  # a probe that cannot answer must not break training
        return None


def measure_batch_bytes(obj):
    """Total bytes of every tensor reachable in a collated batch."""
    import torch

    total = 0
    stack = [obj]
    while stack:
        cur = stack.pop()
        if torch.is_tensor(cur):
            total += cur.element_size() * cur.nelement()
        elif isinstance(cur, dict):
            stack.extend(cur.values())
        elif isinstance(cur, (list, tuple)):
            stack.extend(cur)
    return total


def probe_batch_bytes(dataset, collate_fn, logger=None, **loader_kwargs):
    """Load ONE batch in-process and measure it. Returns bytes, or None.

    ``loader_kwargs`` carries whatever the real loader batches by -- rvc passes
    ``batch_sampler=<bucket sampler>``, sovits passes ``batch_size=N``. It is
    deliberately NOT normalized here: the measurement is only worth anything if
    the probe batches the same way the real loader does.

    ★ THE RNG DISCIPLINE IS THE POINT OF THIS FUNCTION, not the measurement.
    ``sovits/data_utils.py`` draws from the GLOBAL ``random`` module inside
    ``__getitem__`` (volume augmentation ``random.choice``/``random.uniform`` and a
    ``random.randint`` spec crop). Reading one extra batch would therefore shift
    the global stream and change every subsequent training step -- a silent
    trajectory change that the loss-trajectory parity gate would catch only if
    someone happened to re-run it. (rvc's ``__getitem__`` draws nothing; the
    bucket sampler is safe either way, it seeds a LOCAL generator from the epoch.)

    So all three global streams are snapshotted and restored, and the restore is
    ASSERTED rather than assumed. If it cannot be verified the function reports
    no measurement and the caller keeps its defaults: never trade a trajectory
    change for a memory optimization.
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    log = logger or _LOG
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        probe = DataLoader(
            dataset,
            num_workers=0,  # in-process on purpose: no file mappings, no spawn
            collate_fn=collate_fn,
            **loader_kwargs,
        )
        batch = next(iter(probe))
        size = measure_batch_bytes(batch)
        del batch, probe
    except Exception as e:
        log.warning("loader budget: batch probe failed (%s: %s) - keeping defaults", type(e).__name__, e)
        size = None
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)

    # self-check: prove the restore actually happened, do not assume it
    if random.getstate() != py_state or not torch.equal(torch.get_rng_state(), torch_state):
        log.error("loader budget: RNG restore FAILED after the probe - keeping defaults and NOT trusting the measurement")
        return None
    return size


def plan_loader(requested_workers, requested_prefetch, batch_bytes, available_bytes,
                fraction=BUDGET_FRACTION, logger=None):
    """Return ``(num_workers, prefetch_factor)``. REDUCE-ONLY -- never raises either.

    Ladder, most-throughput-preserving first:
      1. requested numbers fit         -> unchanged (identical to today's behaviour)
      2. drop prefetch toward MIN_PREFETCH (queue depth costs latency hiding only)
      3. drop workers toward 1        (each worker lost costs parallel decode)
      4. 0 workers                    (in-process: zero shared mappings, cannot 1455)
    """
    log = logger or _LOG
    w, p = int(requested_workers), int(requested_prefetch)
    # ⚠ S115: ``available_bytes`` is tested for None, NOT for falsiness. A MEASURED zero is
    # the most dangerous state this whole module exists for -- the commit charge is spent,
    # which IS the Windows-1455 condition -- and ``not available_bytes`` used to send it down
    # the "change nothing" path, granting the full request with the guard silent. A measured
    # zero now walks the ladder and lands on 0 workers (in-process: no shared mappings, so
    # 1455 is structurally impossible). Only None -- "the probe could not answer" -- means
    # keep the defaults.
    # ``batch_bytes`` keeps the falsy test on purpose: it is the DIVISOR of ``budget`` below,
    # and a batch measured at 0 bytes bounds nothing anyway, so 0 and None are genuinely the
    # same answer there. The asymmetry is the point; do not "tidy" the two into one form.
    if w <= 0 or not batch_bytes or available_bytes is None:
        log.info(
            "loader budget: no adaptation (workers=%s prefetch=%s batch_bytes=%s avail_commit=%s)",
            w, p, batch_bytes, available_bytes,
        )
        return w, p

    budget = int(available_bytes * fraction)
    fits = budget // batch_bytes  # how many in-flight batches the budget allows
    want = w * p

    if fits >= want:
        log.info(
            "loader budget: keeping workers=%d prefetch=%d (%d batches x %.1f MiB = %.0f MiB "
            "fits in %.0f MiB = %.0f%% of %.0f MiB available commit)",
            w, p, want, batch_bytes / 1048576, want * batch_bytes / 1048576,
            budget / 1048576, fraction * 100, available_bytes / 1048576,
        )
        return w, p

    original = (w, p)
    while w * p > fits and p > MIN_PREFETCH:
        p -= 1
    while w * p > fits and w > 1:
        w -= 1
    if w * p > fits:
        w, p = 0, requested_prefetch  # prefetch_factor is ignored when workers == 0

    log.warning(
        "loader budget: REDUCING workers %d->%d prefetch %d->%d - %d in-flight batches x "
        "%.1f MiB would need %.0f MiB but only %.0f MiB (%.0f%% of %.0f MiB available commit) "
        "is budgeted. This is the guard against the 'training froze' report (Windows error 1455, "
        "paging file too small). More headroom: raise the pagefile, close other GPU/render work, "
        "or use a smaller batch size.",
        original[0], w, original[1], p, want, batch_bytes / 1048576,
        want * batch_bytes / 1048576, budget / 1048576, fraction * 100,
        available_bytes / 1048576,
    )
    return w, p
