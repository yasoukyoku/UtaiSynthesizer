# Vendored from so-vits-svc 4.0-v2 (modules/audio.py @ cf5a8fb) — the
# "aam" mel recipe: librosa stft (center=True) -> mel (80, fmin 0, fmax 22050)
# -> amp_to_db (min_level_db -115) - ref_level_db (20) -> normalize to [0, 4].
# This is the POSTERIOR/mel_am target mel, distinct from (1) the torch loss mel
# (modules/mel_processing.py), (2) the 2048 linear spec recipe, and (3) the
# diffusion nsf_hifigan 128-mel — four recipes total, never mix them.
# Changes vs upstream (deliberate, zero-math — gate0 compares against the
# original running on librosa 0.9.1):
#   - librosa >=0.10 API: filters.mel/resample take keyword-only args
#     (upstream's 0.8 positional call), librosa.core.load -> librosa.load.
#   - stft pad_mode pinned to 'constant'. Upstream's own requirements SPLIT
#     here: requirements.txt pins librosa 0.8.1 (stft default 'reflect') while
#     requirements_win.txt pins 0.9.2 (default 'constant', changed in 0.9.0) —
#     upstream Windows and Linux users trained with DIFFERENT edge-frame mels.
#     We pin the Windows/0.9.x lineage: it matches the project's "era
#     reference environment" (RVC runtime, librosa 0.9.1) that every training
#     gate uses as ground truth, and gate0 C4 verified it empirically against
#     that oracle (constant: 5.4e-7 = FFT noise; reflect: 1.09 at edge
#     frames). Difference is edge-frames-only either way; pinned explicitly so
#     a future librosa default flip can never drift the recipe.
#   - dead upstream helpers dropped: save_wav / _mel_to_linear / _db_to_amp /
#     _istft / duplicated _stft definition (nothing in the training chain calls
#     them; load_wav lives in utils.py like upstream's live import path).
import numpy as np
import librosa
import librosa.filters


_mel_basis = None


def _build_mel_basis(hparams):
    assert hparams.fmax <= hparams.sampling_rate // 2
    return librosa.filters.mel(sr=hparams.sampling_rate,
                               n_fft=hparams.n_fft,
                               n_mels=hparams.acoustic_dim,
                               fmin=hparams.fmin,
                               fmax=hparams.fmax)


def _linear_to_mel(spectogram, hparams):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(hparams)
    return np.dot(_mel_basis, spectogram)


def _stft(y, hparams):
    # pad_mode pinned to the empirically-verified 0.9.1-era behavior (header)
    return librosa.stft(y=y,
                        n_fft=hparams.n_fft,
                        hop_length=hparams.hop_length,
                        win_length=hparams.win_size,
                        pad_mode="constant")


def _amp_to_db(x, hparams):
    min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _normalize(S, hparams):
    return hparams.max_abs_value * np.clip(((S - hparams.min_db) /
                                            (-hparams.min_db)), 0, 1)


def melspectrogram(wav, hparams):
    D = _stft(wav, hparams)
    S = _amp_to_db(_linear_to_mel(np.abs(D), hparams),
                   hparams) - hparams.ref_level_db
    return _normalize(S, hparams)
