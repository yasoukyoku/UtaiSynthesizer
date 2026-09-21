# Vendored verbatim from so-vits-svc 4.1-Stable (modules/F0Predictor/rmvpe/inference.py @ 730930d).
# Changes vs upstream: package-relative imports only; S172 routes the network forward through
# the MIOpen guard (see mel2hidden) — a no-op off ROCm, and on ROCm it is what keeps this lane
# alive when the run-time kernel compiler cannot build MIOpen's BatchNorm/RNN kernels.
import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample

from .... import miopen_guard
from .constants import *  # noqa: F403
from .model import E2E0
from .spec import MelSpectrogram
from .utils import to_local_average_cents, to_viterbi_cents


class RMVPE:
    def __init__(self, model_path, device=None, dtype = torch.float32, hop_length=160):
        self.resample_kernel = {}
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        model = E2E0(4, 1, (2, 2))
        ckpt = torch.load(model_path, map_location=torch.device(self.device), weights_only=False)
        model.load_state_dict(ckpt['model'])
        model = model.to(dtype).to(self.device)
        model.eval()
        self.model = model
        self.dtype = dtype
        self.mel_extractor = MelSpectrogram(N_MELS, SAMPLE_RATE, WINDOW_LENGTH, hop_length, None, MEL_FMIN, MEL_FMAX)  # noqa: F405
        self.resample_kernel = {}

    def mel2hidden(self, mel):
        with torch.no_grad():
            n_frames = mel.shape[-1]
            mel = F.pad(mel, (0, 32 * ((n_frames - 1) // 32 + 1) - n_frames), mode='constant')
            # S172 deviation: the WHOLE network forward goes through the MIOpen guard, not
            # just the BiGRU inside it. This lane runs fp32 (sovits/extract.py passes
            # dtype=torch.float32), and in fp32 torch hands BatchNorm to MIOpen too — so on a
            # machine whose run-time kernel compiler has no C++ headers this dies in
            # deepunet.py's `self.bn(x)`, BEFORE it ever reaches the RNN. Guarding only the
            # RNN leaves this lane completely unprotected (measured: identical traceback).
            hidden = miopen_guard.run(lambda: self.model(mel))
            return hidden[:, :n_frames]

    def decode(self, hidden, thred=0.03, use_viterbi=False):
        if use_viterbi:
            cents_pred = to_viterbi_cents(hidden, thred=thred)
        else:
            cents_pred = to_local_average_cents(hidden, thred=thred)
        f0 = torch.Tensor([10 * (2 ** (cent_pred / 1200)) if cent_pred else 0 for cent_pred in cents_pred]).to(self.device)
        return f0

    def infer_from_audio(self, audio, sample_rate=16000, thred=0.05, use_viterbi=False):
        audio = audio.unsqueeze(0).to(self.dtype).to(self.device)
        if sample_rate == 16000:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, 16000, lowpass_filter_width=128)
            self.resample_kernel[key_str] = self.resample_kernel[key_str].to(self.dtype).to(self.device)
            audio_res = self.resample_kernel[key_str](audio)
        mel_extractor = self.mel_extractor.to(self.device)
        mel = mel_extractor(audio_res, center=True).to(self.dtype)
        hidden = self.mel2hidden(mel)
        f0 = self.decode(hidden.squeeze(0), thred=thred, use_viterbi=use_viterbi)
        return f0
