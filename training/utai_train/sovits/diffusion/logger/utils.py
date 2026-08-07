# Vendored from so-vits-svc 4.1-Stable diffusion/logger/utils.py (@ 730930d).
# load_model (the resume/base-model scanner: max model_*.pt step wins,
# strict=False load, optional optimizer state) is semantically UNCHANGED.
# Deviations: print() -> logging; load_config/save_config read/write UTF-8
# (locale-codec writes mojibake CJK display names on Chinese/Japanese Windows).
#   - S118 §F8⒜: load_model SPLIT into latest_numbered_path (the scan) +
#     load_checkpoint_file (the load) and rebuilt out of them, so the resume
#     chooser can ask WHICH file without loading it and both paths read a
#     checkpoint through one implementation. The arithmetic, the recursion, the
#     non-digit-means-0 rule and the unreachable best.pt branch are verbatim.
#     load_checkpoint_file additionally REPORTS whether the file carried an
#     optimizer and LOGS strict=False's _IncompatibleKeys when non-empty —
#     both were previously discarded in silence.
import json
import logging
import os

import torch
import yaml

logger = logging.getLogger(__name__)


def traverse_dir(
        root_dir,
        extensions,
        amount=None,
        str_include=None,
        str_exclude=None,
        is_pure=False,
        is_sort=False,
        is_ext=True):

    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if any([file.endswith(f".{ext}") for ext in extensions]):
                # path
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path

                # amount
                if (amount is not None) and (cnt == amount):
                    if is_sort:
                        file_list.sort()
                    return file_list
                
                # check string
                if (str_include is not None) and (str_include not in pure_path):
                    continue
                if (str_exclude is not None) and (str_exclude in pure_path):
                    continue
                
                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext)+1)]
                file_list.append(pure_path)
                cnt += 1
    if is_sort:
        file_list.sort()
    return file_list


    
class DotDict(dict):
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   

    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__


def get_network_paras_amount(model_dict):
    info = dict()
    for model_name, model in model_dict.items():
        # all_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        info[model_name] = trainable_params
    return info


def load_config(path_config):
    with open(path_config, "r", encoding="utf-8") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    # print(args)
    return args

def save_config(path_config,config):
    config = dict(config)
    with open(path_config, "w", encoding="utf-8") as f:
        yaml.dump(config, f)

def to_json(path_params, path_json):
    params = torch.load(path_params, map_location=torch.device('cpu'), weights_only=False)
    raw_state_dict = {}
    for k, v in params.items():
        val = v.flatten().numpy().tolist()
        raw_state_dict[k] = val

    with open(path_json, 'w') as outfile:
        json.dump(raw_state_dict, outfile,indent= "\t")


def convert_tensor_to_numpy(tensor, is_squeeze=True):
    if is_squeeze:
        tensor = tensor.squeeze()
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.numpy()

           
def latest_numbered_path(expdir, name='model', postfix=''):
    """Which checkpoint upstream's resume scan picks. Returns ``(path, step)``, or ``(None, 0)``.

    ★S118: extracted from ``load_model`` VERBATIM — same recursive ``traverse_dir``, same string
    slicing, same ``max`` with the non-digit-means-0 fallback, and the unreachable ``best.pt``
    branch kept rather than tidied away. Its reason for existing is that S118's resume chooser
    has to COMPARE this step against the one in a ``resume_latest`` snapshot, and it must not pay
    a 210-630 MB ``torch.load`` to find it out.

    ⚠ Two properties of this arithmetic are load-bearing elsewhere and were measured, not read
    (`TESTING/s118_f8a/probe_nested_pt_arithmetic.py`):
      * the returned path is REBUILT from ``str(maxstep)``, never taken from the scan, so a
        nested ``.pt`` can never be selected — but it can still raise ``maxstep`` if its
        directory name is short enough to be consumed by the slice (see
        ``resume_state.SNAPSHOT_DIR_MIN_LEN``);
      * the path may therefore NOT EXIST (``model_0100.pt`` -> ``model_100.pt``, or a hijacked
        step). Callers must check; ``load_model`` keeps upstream's behaviour of letting
        ``torch.load`` raise.
    """
    if postfix == '':
        postfix = '_' + postfix
    path = os.path.join(expdir, name+postfix)
    path_pt = traverse_dir(expdir, ['pt'], is_ext=False)
    if len(path_pt) == 0:
        return None, 0
    steps = [s[len(path):] for s in path_pt]
    maxstep = max([int(s) if s.isdigit() else 0 for s in steps])
    if maxstep >= 0:
        return path+str(maxstep)+'.pt', maxstep
    # Unreachable: max() over non-negative ints. Kept because it is upstream's, and because its
    # existence is what makes the file LOOK like it has a best-checkpoint fallback when it does
    # not (measured: an expdir holding only model_best.pt dies with FileNotFoundError).
    return path+'best.pt', 0


def load_checkpoint_file(path_pt, model, optimizer, device='cpu'):
    """Restore ONE checkpoint file into ``model``/``optimizer``. Returns ``(step, had_optimizer)``.

    Semantically the body of upstream's ``load_model`` after it picked a file. Two deviations,
    both "say what happened", neither touching the math:

    * ``had_optimizer`` is REPORTED instead of swallowed. ``diffusion_template.yaml`` ships
      upstream's ``save_opt: false``, so every periodic checkpoint is ``{global_step, model}``
      and this branch simply does not run — measured, the AdamW state stays ``{}`` and not one
      line was logged. The caller decides what to say about that (S118 §F8⒜).
    * the ``_IncompatibleKeys`` that ``strict=False`` returns is LOGGED when non-empty instead of
      discarded. ``strict=False`` is upstream's and stays (a base model legitimately need not
      match key-for-key, and a SIZE mismatch still raises), but a checkpoint that quietly leaves
      parameters at their constructor values is the class of defect ``ckpt_guard`` exists for on
      the three GAN trainers. ⚠ Measured to be silent on the healthy path: our real vec768
      ``model_0.pt`` and a real trained ``model_180.pt`` both report 0 missing / 0 unexpected.
    """
    logger.info(' [*] restoring model from %s', path_pt)
    ckpt = torch.load(path_pt, map_location=torch.device(device), weights_only=False)
    incompatible = model.load_state_dict(ckpt['model'], strict=False)
    missing = list(getattr(incompatible, 'missing_keys', ()) or ())
    unexpected = list(getattr(incompatible, 'unexpected_keys', ()) or ())
    if missing or unexpected:
        logger.warning(
            ' [!] checkpoint does not match this model: %d parameter(s) MISSING from it '
            '(they keep their freshly-initialised values: %s) and %d unexpected (%s)',
            len(missing), ", ".join(missing[:6]) or "-",
            len(unexpected), ", ".join(unexpected[:6]) or "-",
        )
    had_optimizer = ckpt.get("optimizer") is not None
    if had_optimizer:
        optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt['global_step'], had_optimizer


def load_model(
        expdir,
        model,
        optimizer,
        name='model',
        postfix='',
        device='cpu'):
    """Upstream's resume entry point — behaviour UNCHANGED, now expressed as scan + load.

    ⚠ Production no longer calls this: `diff_pipeline.load_start_state` does, because the choice
    between the numbered grid and S118's snapshots cannot be made inside a function whose whole
    interface is "pick the highest number". It stays because it is the vendored shape the
    upstream loss-trajectory gate drives, and it is built out of the SAME two helpers so the two
    paths can never start disagreeing about how a checkpoint is read.
    """
    global_step = 0
    path_pt, _ = latest_numbered_path(expdir, name, postfix)
    if path_pt is not None:
        global_step, _ = load_checkpoint_file(path_pt, model, optimizer, device)
    return global_step, model, optimizer
