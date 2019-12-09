import argparse

import torch
from tqdm import tqdm

from data.cp_data_loader import SpectrogramDataset, AudioDataLoader
from transformer.opts import add_decoder_args, add_inference_args
from transformer.utils import load_model

parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser = add_inference_args(parser)
parser.add_argument('--test-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/test_manifest.csv')
parser.add_argument('--batch-size', default=20, type=int, help='Batch size for testing')
parser.add_argument('--num-workers', default=0, type=int, help='Number of workers used in dataloading')
parser.add_argument('--verbose', action="store_true", help="print out decoded output and error of each sample")
parser.add_argument('--save-output', default=None, help="Saves output of model from test to this file_path")
parser = add_decoder_args(parser)


def evaluate(test_loader, device, model, criterion, decoder, target_decoder, save_output=False, verbose=False,
             half=False):
    model.eval()
    total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
    output_data = []
    avg_loss = 0
    corr = 0

    for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
        inputs, targets, input_percentages = data
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        inputs = inputs.to(device)
        targets = targets.long().to(device)

        if half:
            inputs = inputs.half()
            targets = targets.half()
        # unflatten targets
        split_targets = []

        _, hidden, _ = model(inputs, input_sizes)
        hidden_out = hidden.float()  # ensure float32 for loss

        if save_output is not None:
            # add output to data array, and continue
            output_data.append(hidden)

        loss = criterion(hidden_out, targets).to(device)
        loss = loss / inputs.size(0)  # average the loss by minibatch
        loss_value = loss.item()
        avg_loss += loss_value

        scores, idxs = hidden_out.max(1)
        corr += torch.sum(idxs == targets)

    avg_loss /= len(test_loader)
    return avg_loss, output_data, corr.item()


def evaluate_class(test_loader, device, model):
    model.eval()
    corr = 0
    for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
        inputs, targets, input_percentages = data
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        inputs = inputs.to(device)
        targets = targets.long().to(device)

        out, hidden, output_sizes = model(inputs, input_sizes)
        hidden_out = hidden.float()  # ensure float32 for loss
        scores, idxs = hidden_out.max(1)
        corr += torch.sum(idxs == targets)
    return corr.item()


if __name__ == '__main__':
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if args.cuda else "cpu")
    model = load_model(device, args.model_path, args.half)

    test_dataset = SpectrogramDataset(audio_conf=model.audio_conf, manifest_filepath=args.test_manifest,
                                      labels=model.labels, normalize=True)
    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)
    corr = evaluate_class(test_loader=test_loader,
                          device=device,
                          model=model)

    size = float(len(test_dataset))
    print(corr, size, corr / size)

    # print('Test Summary \t'
    #       'Average WER {wer:.3f}\t'
    #       'Average CER {cer:.3f}\t'.format(wer=wer, cer=cer))
    # if args.save_output is not None:
    #     np.save(args.save_output, output_data)
