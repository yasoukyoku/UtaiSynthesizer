# Vendored verbatim from so-vits-svc 4.1-Stable (modules/F0Predictor/rmvpe/seq.py @ 730930d).
# Changes vs upstream: package-relative imports only; S172 routes the RNN call through the
# MIOpen-RNN backstop (utai_train/miopen_guard.py) — a no-op on every non-ROCm runtime, and on
# ROCm it survives MIOpen failing to BUILD its RNN kernel, which is fatal for f0 extraction on
# any machine whose hipRTC cannot resolve <type_traits> (the pack ships no C++ standard library).
import torch.nn as nn

from .... import miopen_guard


class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(input_features, hidden_features, num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        return miopen_guard.run(lambda: self.gru(x)[0])


class BiLSTM(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_features, hidden_features, num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        return miopen_guard.run(lambda: self.lstm(x)[0])
