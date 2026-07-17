# Vendored from so-vits-svc 4.0-v2 (data_utils.py @ cf5a8fb).
# Training math / RNG stream UNCHANGED: BaseDataset seeds the global `random`
# with train.seed and shuffles the filelist at construction (train dataset
# first, then valid — both reseed), and SingDataset's >550-frame random crop
# draws from that same global stream (num_workers=0) — the loss-trajectory gate
# depends on this exact consumption order.
# Changes vs upstream (deliberate):
#   - package-relative imports; sys.path hack dropped
#   - the lazily-generated aam mel is named `<wav>.aam80.npy` instead of
#     upstream's `<wav>.mel.npy` (zero-math rename: the shallow-diffusion
#     pipeline's nsf_hifigan 128-mel already claims `.mel.npy` in shared
#     workspaces — a name collision would silently poison one of the two);
#     the pipeline pre-computes it in the extract stage, this lazy path is a
#     self-heal fallback producing byte-identical output
#   - torch.load with explicit weights_only=False (torch>=2.6 default flip;
#     S43 convention across utai_train)
#   - dead parse_label() pruned (references an undefined `npu`; nothing in the
#     v2 chain calls it) along with its unused string/json/math imports
#   - DataLoader worker counts read from train.num_workers (default 4 = the
#     upstream hardcode; the determinism gate sets 0 on both sides) and
#     persistent_workers on Windows (per-epoch worker respawn re-imports torch
#     every epoch; math unchanged — only the >550-crop draws of worker
#     processes come from per-worker RNG either way)
import os
import random
import numpy as np
from torch.utils.data import DataLoader
import torch

from . import utils
from .modules import audio
from .utils import load_wav


class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, hparams, fileid_list_path):
        self.hparams = hparams
        self.fileid_list = self.get_fileid_list(fileid_list_path)
        random.seed(hparams.train.seed)
        random.shuffle(self.fileid_list)
        if (hparams.data.n_speakers > 0):
            self.spk2id = hparams.spk

    def get_fileid_list(self, fileid_list_path):
        fileid_list = []
        with open(fileid_list_path, 'r') as f:
            for line in f.readlines():
                fileid_list.append(line.strip())

        return fileid_list

    def __len__(self):
        return len(self.fileid_list)


class SingDataset(BaseDataset):
    def __init__(self, hparams, data_dir, fileid_list_path):
        BaseDataset.__init__(self, hparams, fileid_list_path)
        self.hps = hparams
        self.data_dir = data_dir
        # self.__filter__()

    def __filter__(self):
        new_fileid_list= []
        for wav_path in self.fileid_list:
            # mel_path = wav_path + ".mel.npy"
            # mel = np.load(mel_path)
            # if mel.shape[0] < 60:
            #     print("skip short audio:", wav_path)
            #     continue
            # if mel.shape[0] > 800:
            #     print("skip long audio:", wav_path)
            #     continue
            # assert mel.shape[1] == 80
            new_fileid_list.append(wav_path)
        print("original length:", len(self.fileid_list))
        print("filtered length:", len(new_fileid_list))
        self.fileid_list = new_fileid_list

    def interpolate_f0(self, data):
        '''
        对F0进行插值处理
        '''
        data = np.reshape(data, (data.size, 1))

        vuv_vector = np.zeros((data.size, 1), dtype=np.float32)
        vuv_vector[data > 0.0] = 1.0
        vuv_vector[data <= 0.0] = 0.0

        ip_data = data

        frame_number = data.size
        last_value = 0.0
        for i in range(frame_number):
            if data[i] <= 0.0:
                j = i + 1
                for j in range(i + 1, frame_number):
                    if data[j] > 0.0:
                        break
                if j < frame_number - 1:
                    if last_value > 0.0:
                        step = (data[j] - data[i - 1]) / float(j - i)
                        for k in range(i, j):
                            ip_data[k] = data[i - 1] + step * (k - i + 1)
                    else:
                        for k in range(i, j):
                            ip_data[k] = data[j]
                else:
                    for k in range(i, frame_number):
                        ip_data[k] = last_value
            else:
                ip_data[i] = data[i]
                last_value = data[i]

        return ip_data, vuv_vector

    def __getitem__(self, index):
        wav_path = self.fileid_list[index]

        spk = wav_path.split('/')[-2]
        spkid = self.spk2id[spk]

        wav = load_wav(wav_path,
                       raw_sr=self.hparams.data.sampling_rate,
                       target_sr=self.hparams.data.sampling_rate,
                       win_size=self.hparams.data.win_size,
                       hop_size=self.hparams.data.hop_length)

        # .aam80.npy, not upstream's .mel.npy — see file header (pre-computed by
        # the extract stage; this lazy branch is a byte-identical self-heal)
        mel_path = wav_path + ".aam80.npy"
        if not os.path.exists(mel_path):
            mel = audio.melspectrogram(wav, self.hparams.data).astype(np.float32).T
            np.save(mel_path, mel)
        else:
            mel = np.load(mel_path)

        if mel.shape[0] < 30:
            print("skip short audio:", self.fileid_list[index])
            return None
        assert mel.shape[1] == 80
        mel = torch.FloatTensor(mel).transpose(0, 1)

        f0_path = wav_path + ".f0.npy"
        f0 = np.load(f0_path)
        assert abs(f0.shape[0]-mel.shape[1]) < 2, (f0.shape ,mel.shape)
        sum_dur = min(f0.shape[0], mel.shape[1])
        f0 = f0[:sum_dur]
        mel = mel[:, :sum_dur]

        f0, uv = self.interpolate_f0(f0)
        f0 = f0.reshape([-1])
        f0 = torch.FloatTensor(f0).reshape([1, -1])

        uv = uv.reshape([-1])
        uv = torch.FloatTensor(uv).reshape([1, -1])

        wav = wav.reshape(-1)
        if (wav.shape[0] != sum_dur * self.hparams.data.hop_length):
            if (abs(wav.shape[0] - sum_dur * self.hparams.data.hop_length) > 3 * self.hparams.data.hop_length):
                print("dataset error wav : ", wav.shape, sum_dur)
                return None
            if (wav.shape[0] > sum_dur * self.hparams.data.hop_length):
                wav = wav[:sum_dur * self.hparams.data.hop_length]
            else:
                wav = np.concatenate([wav, np.zeros([sum_dur * self.hparams.data.hop_length - wav.shape[0]])], axis=0)
        wav = torch.FloatTensor(wav).reshape([1, -1])

        c_path = wav_path + ".soft.pt"
        c = torch.load(c_path, weights_only=False)
        c = utils.repeat_expand_2d(c.squeeze(0), sum_dur)

        assert f0.shape[1] == mel.shape[1]

        if mel.shape[1] > 550:
            start = random.randint(0, mel.shape[1]-550)
            end = start + 540
            mel = mel[:, start:end]
            f0 = f0[:, start:end]
            uv = uv[:, start:end]
            c = c[:, start:end]
            wav = wav[:, start*self.hparams.data.hop_length:end*self.hparams.data.hop_length]
        return c, mel, f0, wav, spkid, uv


class SingCollate():

    def __init__(self, hparams):
        self.hparams = hparams
        self.mel_dim = self.hparams.data.acoustic_dim

    def __call__(self, batch):
        batch = [b for b in batch if b is not None]

        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)

        max_c_len = max([x[0].size(1) for x in batch])
        max_mel_len = max([x[1].size(1) for x in batch])
        max_f0_len = max([x[2].size(1) for x in batch])
        max_wav_len = max([x[3].size(1) for x in batch])

        c_lengths = torch.LongTensor(len(batch))
        mel_lengths = torch.LongTensor(len(batch))
        f0_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))

        c_padded = torch.FloatTensor(len(batch), self.hparams.data.c_dim, max_mel_len)
        mel_padded = torch.FloatTensor(len(batch), self.hparams.data.acoustic_dim, max_mel_len)
        f0_padded = torch.FloatTensor(len(batch), 1, max_f0_len)
        uv_padded = torch.FloatTensor(len(batch), 1, max_f0_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        spkids = torch.LongTensor(len(batch))

        c_padded.zero_()
        mel_padded.zero_()
        f0_padded.zero_()
        uv_padded.zero_()
        wav_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            c = row[0]
            c_padded[i, :, :c.size(1)] = c
            c_lengths[i] = c.size(1)

            mel = row[1]
            mel_padded[i, :, :mel.size(1)] = mel
            mel_lengths[i] = mel.size(1)

            f0 = row[2]
            f0_padded[i, :, :f0.size(1)] = f0
            f0_lengths[i] = f0.size(1)

            wav = row[3]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            spkids[i] = row[4]

            uv = row[5]
            uv_padded[i, :, :uv.size(1)] = uv


        data_dict = {}

        data_dict["c"] = c_padded
        data_dict["mel"] = mel_padded
        data_dict["f0"] = f0_padded
        data_dict["uv"] = uv_padded
        data_dict["wav"] = wav_padded

        data_dict["c_lengths"] = c_lengths
        data_dict["mel_lengths"] = mel_lengths
        data_dict["f0_lengths"] = f0_lengths
        data_dict["wav_lengths"] = wav_lengths
        data_dict["spkid"] = spkids

        return data_dict


class DatasetConstructor():

    def __init__(self, hparams, num_replicas=1, rank=1):
        self.hparams = hparams
        self.num_replicas = num_replicas
        self.rank = rank
        self.dataset_function = {"SingDataset": SingDataset}
        self.collate_function = {"SingCollate": SingCollate}
        self._get_components()

    def _get_components(self):
        self._init_datasets()
        self._init_collate()
        self._init_data_loaders()

    def _init_datasets(self):
        self._train_dataset = self.dataset_function[self.hparams.data.dataset_type](self.hparams,
                                                                                    self.hparams.data.data_dir,
                                                                                    self.hparams.data.training_filelist)
        self._valid_dataset = self.dataset_function[self.hparams.data.dataset_type](self.hparams,
                                                                                    self.hparams.data.data_dir,
                                                                                    self.hparams.data.validation_filelist)

    def _init_collate(self):
        self._collate_fn = self.collate_function[self.hparams.data.collate_type](self.hparams)

    def _init_data_loaders(self):
        train_sampler = torch.utils.data.distributed.DistributedSampler(self._train_dataset,
                                                                        num_replicas=self.num_replicas, rank=self.rank,
                                                                        shuffle=True)

        # upstream hardcodes 4/1; train.num_workers (default 4) keeps that
        # while letting the determinism gate pin both loaders to 0
        num_workers = int(getattr(self.hparams.train, "num_workers", 4))
        valid_workers = 1 if num_workers > 0 else 0
        self.train_loader = DataLoader(self._train_dataset, num_workers=num_workers, shuffle=False,
                                       batch_size=self.hparams.train.batch_size, pin_memory=True,
                                       drop_last=True, collate_fn=self._collate_fn, sampler=train_sampler,
                                       persistent_workers=num_workers > 0)

        self.valid_loader = DataLoader(self._valid_dataset, num_workers=valid_workers, shuffle=False,
                                       batch_size=1, pin_memory=True,
                                       drop_last=True, collate_fn=self._collate_fn,
                                       persistent_workers=valid_workers > 0)

    def get_train_loader(self):
        return self.train_loader

    def get_valid_loader(self):
        return self.valid_loader

