import argparse
import warnings

from opts import add_decoder_args, add_inference_args
from utils import load_model

warnings.simplefilter('ignore')

import torch

from data.data_loader import SpectrogramParser


def infer(audio_path, spect_parser, model, device, use_half):
    spect = spect_parser.parse_audio(audio_path).contiguous()
    spect = spect.view(1, 1, spect.size(0), spect.size(1))
    spect = spect.to(device)
    if use_half:
        spect = spect.half()
    input_sizes = torch.IntTensor([spect.size(3)]).int()
    _, hidden, _ = model(spect, input_sizes)
    hidden_out = hidden.float()

    return hidden_out.max(1)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='DeepSpeech transcription')
    arg_parser = add_inference_args(arg_parser)
    arg_parser.add_argument('--audio-path', default='audio.wav',
                            help='Audio file to predict on')
    arg_parser.add_argument('--offsets', dest='offsets', action='store_true', help='Returns time offset information')
    arg_parser = add_decoder_args(arg_parser)
    args = arg_parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    model = load_model(device, args.model_path, args.half)

    spect_parser = SpectrogramParser(model.audio_conf, normalize=True)

    audio_path = '/home/xiao/code/ai_utils/experiment/child_det/audios/8f4d3b65-927e-4722-a794-d8037d6b561b.wav'
    val, idx = infer(
        # audio_path=args.audio_path,
        audio_path=audio_path,
        spect_parser=spect_parser,
        model=model,
        device=device,
        use_half=args.half)
    print('Infer result: {}'.format(idx.item()))
