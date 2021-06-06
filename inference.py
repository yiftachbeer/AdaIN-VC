import argparse

import soundfile as sf
import torch
import torchaudio

from data import Wav2Mel

PRETRAINED_VC_MODEL_PATH = 'pretrained/vc_model.pt'
PRETRAINED_VOCODER_PATH = 'pretrained/vocoder.pt'


def main(
    source: str,
    target: str,
    output: str,
    model_path: str = PRETRAINED_VC_MODEL_PATH,
    vocoder_path: str = PRETRAINED_VOCODER_PATH,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(model_path, map_location=device)
    vocoder = torch.jit.load(vocoder_path, map_location=device)
    wav2mel = Wav2Mel()

    src, src_sr = torchaudio.load(source)
    tgt, tgt_sr = torchaudio.load(target)

    with torch.no_grad():
        src = wav2mel(src, src_sr)[None, :].to(device)
        tgt = wav2mel(tgt, tgt_sr)[None, :].to(device)

    wav = convert_voice(src, tgt, model, vocoder)

    wav = wav[0].data.cpu().numpy()
    sf.write(output, wav, wav2mel.sample_rate)


def convert_voice(src, tgt, model, vocoder):
    with torch.no_grad():
        cvt = model.inference(src, tgt)
        wav = vocoder.generate([cvt.squeeze(0).data.T])
    return wav


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=str)
    parser.add_argument("target", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--model_path", type=str, default=PRETRAINED_VC_MODEL_PATH)
    parser.add_argument("--vocoder_path", type=str, default=PRETRAINED_VOCODER_PATH)
    main(**vars(parser.parse_args()))
