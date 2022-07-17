# -*- coding: utf-8 -*-
# @Author   : jeffcheng
# @Time     : 2021/9/1 - 15:13
# @Reference: a inference script for single audio, heavily base on demo.py and traintest.py
import argparse
import csv
import glob
import librosa
import numpy as np
import os
import sys
import time
import torch
import torchaudio

import matplotlib.pyplot as plt

from collections import defaultdict
from pathlib import Path

torchaudio.set_audio_backend("soundfile")       # switch backend
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
from src.models import ASTModel

# download pretrained model in this directory
os.environ['TORCH_HOME'] = '../pretrained_models'

SR = 16000

def make_features(track_path, mel_bins, target_length=1024):
    (audio, _) = librosa.core.load(track_path, sr=SR, mono=True)
    waveform = torch.from_numpy(audio).unsqueeze(0)

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=SR, use_energy=False,
        window_type='hanning', num_mel_bins=mel_bins, dither=0.0,
        frame_shift=10)

    n_frames = fbank.shape[0]

    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
    return fbank

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser:'
                                                 'python inference --audio_path ./0OxlgIitVig.wav '
                                                 '--model_path ./pretrained_models/audioset_10_10_0.4593.pth')

    parser.add_argument("--model_path", type=str, required=True,
                        help="the trained model you want to test")
    parser.add_argument('--audio_glob_pattern',
                        help='literal string glob pattern that should resolve to list of audio files',
                        type=str, default="/import/c4dm-datasets/PAJAMA/audio/live/*/*/*mp3")

    args = parser.parse_args()
    SR = 16000
    def feature_segments(path, mel_bins, target_length=128):
        def feature_fn(audio, sr):
            waveform = torch.from_numpy(audio).unsqueeze(0)
            fbank = torchaudio.compliance.kaldi.fbank(
                waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                window_type='hanning', num_mel_bins=mel_bins, dither=0.0,
                frame_shift=10)
            n_frames = fbank.shape[0]
            p = target_length - n_frames
            if p > 0:
                m = torch.nn.ZeroPad2d((0, 0, 0, p))
                fbank = m(fbank)
            elif p < 0:
                fbank = fbank[0:target_length, :]
            fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
            return fbank

        (audio, _) = librosa.core.load(path, sr=SR, mono=True)
        feature_segments = [feature_fn(audio[i*SR : (i + 1)*SR + 1000], SR)
                            for i in range(int(len(audio)/SR))]

        return feature_segments


    # 1. Get files to process
    mp3_files = glob.glob(args.audio_glob_pattern)
    feats = feature_segments(mp3_files[0], mel_bins=128)
    input_tdim = feats[0].shape[0]

    # 2. load the best model and the weights
    checkpoint_path = args.model_path
    ast_mdl = ASTModel(label_dim=527, input_tdim=input_tdim, imagenet_pretrain=False, audioset_pretrain=False)
    print(f'[*INFO] load checkpoint: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path)
    audio_model = torch.nn.DataParallel(ast_mdl, device_ids=[0])
    audio_model.load_state_dict(checkpoint)
    audio_model = audio_model.cuda()
    audio_model.eval()     # set the eval model

    batch_size = 32

    # 3. Iterate over files
    for filepath in mp3_files:
        print(f'[*INFO] processing file: {filepath}')
        audio_path = Path(filepath)
        feats = feature_segments(audio_path, mel_bins=128)

        # 4. feed the data feature to model
        n_batches = len(feats) // batch_size
        if n_batches * batch_size < len(feats):
            n_batches += 1

        label_to_values = defaultdict(list)
        for bi in range(n_batches):
            batch = torch.stack(feats[bi*(batch_size):(bi + 1)*batch_size], dim=0)
            batch = batch.cuda()
            with torch.no_grad():
                output = audio_model.forward(batch)
                output = torch.sigmoid(output)

            outputs = torch.unbind(output)
            speech = 0
            clapping = 63
            cheering = 66
            applause = 67
            music= 137
            piano = 153

            for offset, output in enumerate(outputs):
                result_output = output.data.cpu().numpy()
                result_output[music] > result_output[applause]
                sec = bi * batch_size + offset


        plt.legend(loc="upper left")
        output_image_path = Path("plots", audio_path.stem + ".png")
        plt.savefig(output_image_path)
        plt.close()
