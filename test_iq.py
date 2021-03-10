import argparse
from distutils.util import strtobool
import os
import pickle

import torch
from train_iq import TrainIQ
from utils.data_loader import get_loader
from torchvision.transforms import transforms
import pytorch_lightning as pl
import distutils

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224,
                                 scale=(1.00, 1.2),
                                 ratio=(0.75, 1.3333333333333333)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dim", type=int, default=300,
                        help="Embedding dimensionality of the model")
    parser.add_argument("--hidden_dim", type=int, default=300,
                        help="Hidden dimensionality of the model")
    parser.add_argument("--latent_dim", type=int, default=300,
                        help="Size of latent dimension")
    parser.add_argument("--pwffn_dim", type=int, default=600,
                        help="Size of postionwise feedforward network in transformer")
    parser.add_argument("--enc_num_layers", type=int, default=6,
                        help="Number of transformer layers in encoder and decoder")
    parser.add_argument("--dec_num_layers", type=int, default=6,
                        help="Number of transformer layers in encoder and decoder")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of heads in the multi-head attention")
    parser.add_argument("--lr", type=float, default=3e-5,
                        help="Learning rate of the network")
    parser.add_argument("--num_pretraining_steps", type=float, default=12000,
                        help="Number of pretraining steps before turning on latent transformer")
    parser.add_argument("--total_training_steps", type=int, default=100000,
                        help="Total number of training steps for the model")
    parser.add_argument("--full_kl_step", type=int, default=20000,
                        help="Number of steps until KLD is annealed")
    parser.add_argument("--kl_ceiling", type=float, default=0.5)
    parser.add_argument("--aux_ceiling", type=float, default=1.0)
    parser.add_argument("--image_recon_lambda", type=float, default=0.01,
                        help="How much to scale the image reconstruction loss by")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--enable_t_space", dest="enable_t_space", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--adaptive_lr", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--reconstruct_through_response", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--lambda_z', type=float, default=0.001,
                        help='coefficient to be added in front of the kl loss.')
    parser.add_argument('--lambda_t', type=float, default=0.0001,
                        help='coefficient to be added with the type space loss.')
    parser.add_argument('--lambda_z_t', type=float, default=0.001,
                        help='coefficient to be added with the t and z space loss.')
    # Data args
    parser.add_argument("--emb_file", type=str, default="vectors/glove.6B.300d.txt",
                        help="Filepath for pretrained embeddings")
    parser.add_argument("--dataset", type=str,
                        default="data/processed/iq_dataset.hdf5")
    parser.add_argument("--val_dataset", type=str,
                        default="data/processed/iq_val_dataset.hdf5")
    parser.add_argument("--vocab", type=str, default="vocab.pkl")
    parser.add_argument("--use_gpu", type=bool, default=False) # NOTE: Doesn't work as you'd expect. If "use_gpu" is present in args, then this will always evaluate to true.
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--print_note", type=str, default="")
    parser.add_argument("--input_mode", type=str, default="ans")
    parser.add_argument("--val_every", type=int, default=500)

    parser.add_argument("--latent_transformer", type=bool, default=True)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available()
                          and args.use_gpu else 'cpu')
    args = parser.parse_args()
    args.device = device
    args.root_dir = os.getcwd()

    vocab = pickle.load(open(args.vocab, "rb"))
    trainGVT = TrainIQ(vocab, args).load_from_checkpoint("lightning_logs/version_6/checkpoints/N-Step-Checkpoint_epoch=4_global_step=12000.ckpt", vocab=vocab, args=args).to(args.device)
    trainer = pl.Trainer(max_steps=args.total_training_steps, gradient_clip_val=5,
                            val_check_interval=500, limit_val_batches=100, gpus=args.num_gpus)
    test_data_loader = get_loader(os.path.join(os.getcwd(), args.val_dataset), transform, 128, shuffle=False, num_workers=8)
    trainer.test(trainGVT, test_dataloaders=test_data_loader)
