"""MSST YAML config loading + stem-name metadata — ONE source of truth.

Real MSST training yamls contain `!!python/tuple` tags (e.g. freqs_per_bands,
multi_stft_resolutions_window_sizes), which yaml.safe_load rejects with a
ConstructorError. yaml.FullLoader constructs those tags correctly, so every
converter MUST load MSST yamls through load_msst_yaml() below — never
yaml.safe_load. A yaml the user explicitly provided must HARD-FAIL on parse
errors: a silent fallback bakes wrong STFT/chunk params into the exported
model (e.g. a trained hop of 512 silently becoming the 441/256 defaults).
"""

from typing import Optional


def load_msst_yaml(path) -> dict:
    """Parse an MSST yaml config. Returns {} when path is None.

    Raises (never falls back silently) when the file is missing, unparseable,
    or not a mapping — an explicitly-provided config must be honored exactly.
    """
    if path is None:
        return {}
    import yaml  # hard-fail if pyyaml missing: silent fallback would bake wrong params
    try:
        with open(path, "r", encoding="utf-8") as f:
            # FullLoader: required for the `!!python/tuple` tags in real MSST configs.
            data = yaml.load(f, Loader=yaml.FullLoader)
    except OSError as e:
        raise RuntimeError(f"Cannot read MSST yaml config '{path}': {e}") from e
    except yaml.YAMLError as e:
        raise RuntimeError(
            f"Failed to parse MSST yaml config '{path}': {e}\n"
            f"The config was explicitly provided, so refusing to fall back to defaults."
        ) from e
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise RuntimeError(
            f"MSST yaml config '{path}' did not parse to a mapping "
            f"(got {type(data).__name__})"
        )
    return data


def normalize_stem_names(names) -> list:
    """Lowercase stem labels; in a 2-instrument {vocals, other|instrumental}
    family the non-vocals name becomes 'instrumental' (the user-facing term —
    'other' is a training-side artifact)."""
    low = [str(n).lower() for n in names]
    if len(low) == 2 and "vocals" in low:
        partner = next(n for n in low if n != "vocals") if low.count("vocals") == 1 else None
        if partner in ("other", "instrumental"):
            low = ["vocals" if n == "vocals" else "instrumental" for n in low]
    return low


def stem_fields(instruments, target_instrument, num_stems: int) -> dict:
    """Optional model-json fields describing the model's outputs.

    Returns (possibly empty) dict with:
      "stem_names":    lowercase labels of the model's DIRECT outputs, in
                       training order.
      "residual_name": ONLY for num_stems==1 models — the label of the
                       mixture-minus-stem residual.
    Omits fields when no reliable source of names exists.
    """
    fields = {}
    instruments = list(instruments) if instruments else []
    norm = normalize_stem_names(instruments)
    if num_stems == 1:
        if target_instrument is not None:
            low = [str(i).lower() for i in instruments]
            tgt = str(target_instrument).lower()
            if tgt in low:
                idx = low.index(tgt)
                fields["stem_names"] = [norm[idx]]
                residual = [n for j, n in enumerate(norm) if j != idx]
                if len(residual) == 1:
                    fields["residual_name"] = residual[0]
            else:
                fields["stem_names"] = [tgt]
        elif len(instruments) == 1:
            fields["stem_names"] = norm
    elif len(instruments) == num_stems:
        fields["stem_names"] = norm
    return fields


def stem_fields_from_yaml(yaml_config: Optional[dict], num_stems: int) -> dict:
    """stem_fields() from an MSST yaml's training.instruments/target_instrument."""
    training = (yaml_config or {}).get("training") or {}
    return stem_fields(
        training.get("instruments"), training.get("target_instrument"), num_stems
    )
