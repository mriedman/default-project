from torchvision.utils import save_image, make_grid
# Copyright (c) 2021 Rui Shu
import argparse
import numpy as np
import torch
import tqdm
from pprint import pprint
from torchvision import datasets, transforms
import os
from model import *
from trainer import *
import util
from nnencoder import Encoder32
from sklearn.mixture import GaussianMixture

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

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--iter_max',  type=int, default=1000000, help="Number of training iterations")
parser.add_argument('--iter_save', type=int, default=10000,   help="Save model every n iterations")
parser.add_argument('--run',       type=int, default=3,       help="Run ID. In case you want to run replicates")
parser.add_argument('--overwrite', type=int, default=0,       help="Flag for overwriting")
args = parser.parse_args()
layout = [
    ('model={:s}',  'fsvae'),
    ('run={:04d}', args.run)
]


def eval(args):
    nz, lr, betas, eval_size, num_workers = (128, 2e-4, (0.0, 0.9), 0, 4)
    train_dataloader, eval_dataloader = util.get_dataloaders(
        "./beagle", args.im_size, args.batch_size, eval_size, num_workers
    )

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

    # load encoder
    enc = torch.load('enc-100', map_location=torch.device('cpu'))

    pbar = tqdm(train_dataloader)
    for data, _ in pbar:
        reals, z = prepare_data_for_gan(data, nz, torch.device(args.device))
        z1 = enc(reals)
        x=net_g(torch.mean(z1, dim=0).reshape((1,-1)))
        y = torch.mean(z1, dim=0).reshape((1, -1))
        # x = net_g(torch.mean(z1, dim=0).reshape((1,-1)).repeat((10,1))+torch.randn((200,128))/3)
        # save_image(make_grid(x), 'beagle-2.png')
        save_image(make_grid(net_g(y.repeat((100, 1)) + torch.randn((100, 128)) / 10), nrow=10), 'beagle-2.png')
        # save_image(make_grid(net_g(torch.randn((20,128)).repeat((10,1))+torch.randn((200,128))/3),nrow=20), 'try-f3.png')
        m1 = torch.distributions.MultivariateNormal(y, torch.cov(torch.transpose(z1, 1, 0)) + torch.eye(128) * 1e-2)
        save_image(make_grid(net_g(m1.sample((100,)).reshape((100, 128))), nrow=10), 'beagle-2.png')

        gm = GaussianMixture(n_components=5).fit(z1)
        save_image(make_grid(net_g(torch.Tensor(gm.sample(100)[0])), nrow=10), 'beagle-3.png')

        exit(0)
        print()

    z = torch.randn((args.batch_size * 10, 128))
    x = net_g(z)
    x1 = x.reshape((200,3,32,32))
    save_image(make_grid(x1, nrow=20), 'try-1.png')
    x = net_g(enc(x))
    x1 = x.reshape((200, 3, 32, 32))
    save_image(make_grid(x1, nrow=20), 'try-2.png')

if __name__ == "__main__":
    eval(parse_args())