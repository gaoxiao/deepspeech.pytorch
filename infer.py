import argparse
import warnings

from opts import add_decoder_args, add_inference_args
from utils import load_model

warnings.simplefilter('ignore')

import torch

from data.data_loader import SpectrogramParser
import os.path


def decode_results(decoded_output, decoded_offsets):
    results = {
        "output": [],
        "_meta": {
            "acoustic_model": {
                "name": os.path.basename(args.model_path)
            },
            "language_model": {
                "name": os.path.basename(args.lm_path) if args.lm_path else None,
            },
            "decoder": {
                "lm": args.lm_path is not None,
                "alpha": args.alpha if args.lm_path is not None else None,
                "beta": args.beta if args.lm_path is not None else None,
                "type": args.decoder,
            }
        }
    }

    for b in range(len(decoded_output)):
        for pi in range(min(args.top_paths, len(decoded_output[b]))):
            result = {'transcription': decoded_output[b][pi]}
            if args.offsets:
                result['offsets'] = decoded_offsets[b][pi].tolist()
            results['output'].append(result)
    return results


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

    audio_path = 'data/audios/90c7cb3d-3455-4de4-a012-cd1790e3c7b9.wav'
    val, idx = infer(
        # audio_path=args.audio_path,
        audio_path=audio_path,
        spect_parser=spect_parser,
        model=model,
        device=device,
        use_half=args.half)
    print('Infer result: {}'.format(idx.item()))
