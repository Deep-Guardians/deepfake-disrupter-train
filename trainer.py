import os
import argparse
import torch
from utils.train_v1 import train
from utils.dataloader_v1 import create_dataloader
from utils.hparams import HParam

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, default='./data/img_align_celeba/img_align_celeba',
                        help="Data directory for train.")
    parser.add_argument('-i', '--config', type=str, default='',
                        help="yaml file for configuration")
    parser.add_argument('-p', '--checkpoint_path', type=str, default=None, #-p
                        help="path of checkpoint pt file")
    args = parser.parse_args()

    hp = HParam(args.config)
    with open(args.config, 'r', encoding='utf-8') as f:
        # store hparams as string
        hp_str = ''.join(f.readlines())

    chkpt_path = args.checkpoint_path if args.checkpoint_path is not None else None

    trainloader, testloader = create_dataloader(hp, args.data_dir)
    torch.set_num_threads(16)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    train(hp, trainloader, testloader, chkpt_path)
