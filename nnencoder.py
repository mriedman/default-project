import os
import pprint
import argparse

from tqdm import tqdm
import torch
from torchmetrics.image.fid import NoTrainInceptionV3
import torch.optim as optim

import util
from model import *
from trainer import evaluate, prepare_data_for_gan, prepare_data_for_inception


class Encoder32(nn.Module):
    def __init__(self, nz=128, ngf=256, bottom_width=4):
        super().__init__()

        self.fc1 = nn.Linear(3 * 32 * 32, ngf)
        self.fc2 = nn.Linear(ngf, ngf)
        self.fc3 = nn.Linear(ngf, ngf)
        self.fc4 = nn.Linear(ngf, nz)

        torch.nn.init.xavier_uniform(self.fc1.weight)
        torch.nn.init.xavier_uniform(self.fc2.weight)
        torch.nn.init.xavier_uniform(self.fc3.weight)
        torch.nn.init.xavier_uniform(self.fc4.weight)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class EncoderGBlock32(nn.Module):
    def __init__(self, ndf=128):
        super().__init__()

        self.block1 = DBlockOptimized(3, ndf)
        self.block2 = DBlock(ndf, ndf, downsample=True)
        self.block3 = DBlock(ndf, ndf, downsample=False)
        self.block4 = DBlock(ndf, ndf, downsample=False)
        self.l5 = SNLinear(ndf, ndf)
        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.l5.weight.data, 1.0)

    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)
        h = torch.sum(h, dim=(2, 3))
        y = self.l5(h)
        return y

def parse_args():
    r"""
    Parses command line arguments.
    """

    root_dir = os.path.abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(root_dir, "data"),
        help="Path to dataset directory.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default='./out/baseline-32-150k/ckpt/150000.pth',
        help="Path to checkpoint used for evaluation.",
    )
    parser.add_argument(
        "--im_size",
        type=int,
        default=32,
        help=(
            "Images are resized to this resolution. "
            "Models are automatically selected based on resolution."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="Minibatch size used during evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda:0" if torch.cuda.is_available() else "cpu"),
        help="Device to evaluate on.",
    )
    parser.add_argument(
        "--submit",
        default=False,
        action="store_true",
        help="Generate Inception embeddings used for leaderboard submission.",
    )

    return parser.parse_args()


def eval(args):
    r"""
    Evaluates specified checkpoint.
    """

    # Set parameters
    nz, eval_size, num_workers = (
        128,
        4000 if args.submit else 10000,
        4,
    )

    # Configure models
    if args.im_size == 32:
        net_g = Generator32()
        net_d = Discriminator32()
    elif args.im_size == 64:
        net_g = Generator64()
        net_d = Discriminator64()
    else:
        raise NotImplementedError(f"Unsupported image size '{args.im_size}'.")

    # Loads checkpoint
    state_dict = torch.load(args.ckpt_path, map_location=torch.device('cpu'))
    net_g.load_state_dict(state_dict["net_g"])
    net_d.load_state_dict(state_dict["net_d"])

    # Configures eval dataloader
    _, eval_dataloader = util.get_dataloaders(
        args.data_dir, args.im_size, args.batch_size, eval_size, num_workers
    )

    enc = EncoderGBlock32()
    # opt = optim.SGD(enc.parameters(), lr=0.01)
    opt = optim.Adam(net_d.parameters(), 2e-4, (0.0, 0.9))
    loss_func = nn.MSELoss()

    for _ in range(10**4):
        target = torch.randn((args.batch_size, 128))
        z = net_g.forward(target)
        opt.zero_grad()
        out = enc(z)
        loss = loss_func(out, target)
        if _%10 == 0:
            print(_)
            print(loss)
        if _%100 == 0:
            torch.save(enc, f"./enc-gb-{_}")
        loss.backward()
        opt.step()




if __name__ == "__main__":
    eval(parse_args())
