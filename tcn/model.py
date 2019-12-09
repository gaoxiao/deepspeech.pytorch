import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tcn.tcn_model import TemporalConvNet


class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class MaskConv(nn.Module):
    def __init__(self, seq_module):
        """
        Adds padding to the output of the module based on the given lengths. This is to ensure that the
        results of the model do not change when batch sizes change during inference.
        Input needs to be in the shape of (BxCxDxT)
        :param seq_module: The sequential module containing the conv stack.
        """
        super(MaskConv, self).__init__()
        self.seq_module = seq_module

    def forward(self, x, lengths):
        """
        :param x: The input of size BxCxDxT
        :param lengths: The actual length of each sequence in the batch
        :return: Masked output from the module
        """
        for module in self.seq_module:
            x = module(x)
            mask = torch.BoolTensor(x.size()).fill_(0)
            if x.is_cuda:
                mask = mask.cuda()
            for i, length in enumerate(lengths):
                length = length.item()
                if (mask[i].size(2) - length) > 0:
                    mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
            x = x.masked_fill(mask, 0)
        return x, lengths


class InferenceBatchSoftmax(nn.Module):
    def forward(self, input_):
        if not self.training:
            return F.softmax(input_, dim=-1)
        else:
            return input_


class DeepSpeech(nn.Module):
    def __init__(self, labels="abc", hidden_size=2048, audio_conf=None, dropout=0.5,
                 ksize=7, levels=8, nhid=450, fc_layers=2):
        super(DeepSpeech, self).__init__()

        # model metadata needed for serialization/deserialization
        if audio_conf is None:
            audio_conf = {}
        self.version = '0.0.1'
        self.hidden_size = hidden_size
        self.audio_conf = audio_conf or {}
        self.labels = labels
        self.dropout = 0.5
        self.fc_layers = fc_layers
        self.ksize = ksize
        self.levels = levels
        self.nhid = nhid

        sample_rate = self.audio_conf.get("sample_rate", 16000)
        window_size = self.audio_conf.get("window_size", 0.02)
        num_classes = len(self.labels)

        self.conv = MaskConv(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            torch.nn.Dropout(self.dropout)
        ))
        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        rnn_input_size = int(math.floor((sample_rate * window_size) / 2) + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 20 - 41) / 2 + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
        rnn_input_size *= 32

        channel_sizes = [self.nhid] * self.levels
        # channel_sizes = [450] * 8
        self.tcn = TemporalConvNet(rnn_input_size, channel_sizes, kernel_size=self.ksize, dropout=dropout)

        h_size = channel_sizes[-1]
        if fc_layers == 0:
            self.fc = nn.Sequential(
                torch.nn.Dropout(self.dropout),
                nn.BatchNorm1d(h_size),
                nn.Linear(h_size, num_classes, bias=False),
            )
        else:
            layers = [
                torch.nn.Dropout(self.dropout),
                nn.BatchNorm1d(h_size),
                nn.Linear(h_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU()]
            for i in range(fc_layers - 1):
                layers.extend([nn.Linear(hidden_size, hidden_size),
                               nn.BatchNorm1d(hidden_size),
                               nn.ReLU()])
            layers.append(nn.Linear(hidden_size, num_classes, bias=False))

            self.fc = nn.Sequential(*layers)

        self.inference_softmax = InferenceBatchSoftmax()

    def forward(self, x, lengths):
        lengths = lengths.cpu().int()
        output_lengths = self.get_seq_lens(lengths)
        max_length = output_lengths.max()
        # Calculate bool mask.
        src_key_padding_mask = torch.arange(max_length).expand(len(output_lengths),
                                                               max_length) > output_lengths.unsqueeze(1)
        src_key_padding_mask = src_key_padding_mask.to(x).bool()

        x, _ = self.conv(x, output_lengths)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension, NxHxT

        output = self.tcn(x)
        output = output[:, :, -1]

        output = self.fc(output)
        output = self.inference_softmax(output)
        return None, output, None

    def get_seq_lens(self, input_length):
        """
        Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
        containing the size sequences that will be output by the network.
        :param input_length: 1D Tensor
        :return: 1D Tensor scaled by model
        """
        seq_len = input_length
        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) / m.stride[1] + 1)
        return seq_len.int()

    @classmethod
    def load_model(cls, path):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls(hidden_size=package['hidden_size'],
                    labels=package['labels'],
                    dropout=package['dropout'],
                    fc_layers=package['fc_layers'],
                    audio_conf=package['audio_conf'])
        model.load_state_dict(package['state_dict'])
        return model

    @classmethod
    def load_model_package(cls, package):
        model = cls(hidden_size=package['hidden_size'],
                    labels=package['labels'],
                    dropout=package['dropout'],
                    fc_layers=package['fc_layers'],
                    audio_conf=package['audio_conf'])
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def serialize(model, optimizer=None, epoch=None, iteration=None, loss_results=None,
                  val_loss_results=None, avg_loss=None, meta=None):
        package = {
            'version': model.version,
            'hidden_size': model.hidden_size,
            'audio_conf': model.audio_conf,
            'labels': model.labels,
            'dropout': model.dropout,
            'fc_layers': model.fc_layers,
            'state_dict': model.state_dict(),
        }
        if optimizer is not None:
            package['optim_dict'] = optimizer.state_dict()
        if avg_loss is not None:
            package['avg_loss'] = avg_loss
        if epoch is not None:
            package['epoch'] = epoch + 1  # increment for readability
        if iteration is not None:
            package['iteration'] = iteration
        if loss_results is not None:
            package['loss_results'] = loss_results
            package['val_loss_results'] = val_loss_results
        if meta is not None:
            package['meta'] = meta
        return package

    @staticmethod
    def get_param_size(model):
        params = 0
        for n, p in model.named_parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp
            print('{}, shape: {}, size: {}'.format(n, p.size(), tmp))
        return params


if __name__ == '__main__':
    import os.path
    import argparse

    parser = argparse.ArgumentParser(description='DeepSpeech model information')
    parser.add_argument('--model-path', default='models/deepspeech_final.pth',
                        help='Path to model file created by training')
    args = parser.parse_args()
    package = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    model = DeepSpeech.load_model(args.model_path)

    print("Model name:         ", os.path.basename(args.model_path))
    print("DeepSpeech version: ", model.version)
    print("")
    print("Recurrent Neural Network Properties")
    print("  Hidden Layers:       ", model.hidden_layers)
    print("  Hidden Size:         ", model.hidden_size)
    print("  Classes:          ", len(model.labels))
    print("")
    print("Model Features")
    print("  Labels:           ", model.labels)
    print("  Sample Rate:      ", model.audio_conf.get("sample_rate", "n/a"))
    print("  Window Type:      ", model.audio_conf.get("window", "n/a"))
    print("  Window Size:      ", model.audio_conf.get("window_size", "n/a"))
    print("  Window Stride:    ", model.audio_conf.get("window_stride", "n/a"))

    if package.get('loss_results', None) is not None:
        print("")
        print("Training Information")
        epochs = package['epoch']
        print("  Epochs:           ", epochs)
        print("  Current Loss:      {0:.3f}".format(package['loss_results'][epochs - 1]))
        print("  Current CER:       {0:.3f}".format(package['cer_results'][epochs - 1]))
        print("  Current WER:       {0:.3f}".format(package['wer_results'][epochs - 1]))
