import datetime
import os

import torch


def to_np(x):
    return x.cpu().numpy()


class VisdomLogger(object):
    def __init__(self, id, num_epochs):
        from visdom import Visdom
        self.viz = Visdom()
        self.opts = dict(title=id, ylabel='', xlabel='Epoch', legend=['Loss', 'WER', 'CER'])
        self.viz_window = None
        self.epochs = torch.arange(1, num_epochs + 1)
        self.visdom_plotter = True

    def update(self, epoch, values):
        x_axis = self.epochs[0:epoch + 1]
        y_axis = torch.stack((values["loss_results"][:epoch + 1],
                              values["wer_results"][:epoch + 1],
                              values["cer_results"][:epoch + 1]),
                             dim=1)
        self.viz_window = self.viz.line(
            X=x_axis,
            Y=y_axis,
            opts=self.opts,
            win=self.viz_window,
            update='replace' if self.viz_window else None
        )

    def load_previous_values(self, start_epoch, package):
        self.update(start_epoch - 1, package)  # Add all values except the iteration we're starting from


class TensorBoardLogger(object):
    def __init__(self, id, log_dir, log_params):
        # dt_string = datetime.datetime.now().strftime("%m-%d_%H:%M")
        log_dir = os.path.join(log_dir, id)
        os.makedirs(log_dir, exist_ok=True)
        from tensorboardX import SummaryWriter
        self.id = id
        self.tensorboard_writer = SummaryWriter(log_dir)
        self.log_params = log_params

    def update_loss(self, epoch, values, parameters=None):
        loss, val_loss = values["loss_results"][epoch], values["val_loss_results"][epoch]
        values = {
            'Avg Train Loss': loss,
            'Avg Val Loss': val_loss,
        }
        self.tensorboard_writer.add_scalars('Epoch Loss', values, epoch + 1)
        if self.log_params:
            for tag, value in parameters():
                tag = tag.replace('.', '/')
                self.tensorboard_writer.add_histogram(tag, to_np(value), epoch + 1)
                self.tensorboard_writer.add_histogram(tag + '/grad', to_np(value.grad), epoch + 1)
        self.tensorboard_writer.flush()

    def log_step(self, name, step, values):
        self.tensorboard_writer.add_scalars(name, values, step)
        self.tensorboard_writer.flush()

    def load_previous_values(self, start_epoch, values):
        loss_results = values["loss_results"][:start_epoch]
        val_loss_results = values["val_loss_results"][:start_epoch]

        for i in range(start_epoch):
            values = {
                'Avg Train Loss': loss_results[i],
                'Avg Val Loss': val_loss_results[i],
            }
            self.tensorboard_writer.add_scalars(self.id, values, i + 1)
