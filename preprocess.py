import argparse
import json
import os
from functools import partial
from pathlib import Path

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torchaudio
from torch import Tensor
from tqdm.auto import tqdm

from data.wav2mel import Wav2Mel


def process_files(audio_file: str, wav2mel: nn.Module) -> Tensor:
    speech_tensor, sample_rate = torchaudio.load(audio_file)
    mel_tensor = wav2mel(speech_tensor, sample_rate)

    return mel_tensor


def main(data_dir: str, save_dir: str, segment: int):
    mp.set_sharing_strategy("file_system")
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    wav2mel = Wav2Mel()
    file2mel = partial(process_files, wav2mel=wav2mel)

    meta_data = {}
    speakers = sorted(os.listdir(data_dir))

    for spk in tqdm(speakers):
        meta_data[spk] = []
        spk_dir = Path(data_dir) / spk
        for wav_file in spk_dir.rglob('*mic2.flac'):
            mel = file2mel(wav_file)
            if mel is not None and mel.shape[-1] > segment:
                torch.save(mel, save_path / wav_file.name)
                meta_data[spk].append(wav_file.name)

    with open(save_path / 'metadata.json', "w") as f:
        json.dump(meta_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("save_dir", type=str)
    parser.add_argument("--segment", type=int, default=128)
    main(**vars(parser.parse_args()))
