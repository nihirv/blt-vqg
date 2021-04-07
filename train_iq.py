import argparse
import copy
from operator import itemgetter
import os
import pickle
from types import SimpleNamespace
from typing import List, OrderedDict
from fairseq.models import transformer
from pytorch_lightning import callbacks
from utils.TextGenerationEvaluationMetrics.multiset_distances import MultisetDistances
from utils.TextGenerationEvaluationMetrics.bert_distances import FBD, EMBD
from torch._C import device
from utils.vocab import build_vocab, load_vocab
from utils.data_loader import get_loader
from utils import NLGEval
from torchvision.transforms import transforms
from copy import deepcopy
from models import IQ
import math
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch.multiprocessing
import math as m
import json
import random
from utils import Lamb
from distutils.util import strtobool


torch.multiprocessing.set_sharing_strategy('file_system')
torch.autograd.set_detect_anomaly(True)


class TrainIQ(pl.LightningModule):
    def __init__(self, vocab, args):
        super().__init__()

        self.latent_transformer = False
        self.vocab = vocab
        self.args = args
        self.hp_string = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}. {}".format(
            args.input_mode, args.emb_dim, "True", args.hidden_dim, args.latent_dim, args.pwffn_dim, args.num_layers, args.num_heads, args.lr, args.batch_size, args.print_note
        )
        self.print_string = "{}_\nCodename: {}".format(
            args.print_note, args.variant)

        self.iter = 0
        self.kliter = 0
        self.nlge = NLGEval(no_glove=True, no_skipthoughts=True)
        metrics = {
            "loss": [],
            "img": [],
            "ppl": [],
            "kld": [],
            "aux": [],
            "elbo": [],
            "rec": [],
        }
        self.val_metrics = deepcopy(metrics)
        self.bleus = []
        self.msjs = []
        self.fbds = []

        self.test_scores = {}

        self.cat2name = sorted(
            json.load(open("data/processed/cat2name.json", "r")))

        self.model = IQ(self.latent_transformer, vocab, args)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.vocab.word2idx[self.vocab.SYM_PAD])
        self.image_recon_criterion = nn.MSELoss()

    def token_decode(self, tokenized_tensor_of_ints, sample=5):
        for i, batch_item in enumerate(tokenized_tensor_of_ints):
            if i == sample:
                break
            sentence_string = " ".join(
                [self.vocab.idx2word[token.item()] for token in batch_item])
            print(sentence_string)
        print()

    def forward(self, batch):
        images, _, questions, posteriors, answers, answer_type_orig, answer_types, answer_types_for_input, _, rcnn_features, rcnn_locations = batch.values()
        images, questions, posteriors, answers, answer_type_orig, answer_types, answer_types_for_input, rcnn_features, rcnn_locations = images.cuda(
        ), questions.to(self.args.device), posteriors.to(self.args.device), answers.to(self.args.device), answer_type_orig.to(self.args.device), answer_types.to(self.args.device), answer_types_for_input.to(self.args.device), rcnn_features.to(self.args.device), rcnn_locations.to(self.args.device)

        answer_input = answers  # answers has answer_type/the category pre-pended to it
        if self.args.variant == "transformer-c":
            answer_input = answer_types_for_input

        output, z_logits, kld_loss, image_recon = self.model(
            images, answer_type_orig, answer_input, posteriors, questions, rcnn_features, rcnn_locations)
        # if random.random() > 0.5:
        #     output, z_logits, kld_loss, image_recon = self.model(
        #         images, answers, posteriors, questions, rcnn_features, rcnn_locations)
        # else:
        #     output, z_logits, kld_loss, image_recon = self.model(
        #         images, answer_types_for_input, posteriors, questions, rcnn_features, rcnn_locations)

        return output, z_logits, kld_loss, image_recon

    def calculate_losses(self, output, image_recon, kld_loss, z_logit, target):
        loss_rec = self.criterion(
            output[1:].reshape(-1, output.size(-1)), target[1:].reshape(-1))  # TODO: Remove [1:] for TF decoder
        # loss_img = self.image_recon_criterion(image_recon[0], image_recon[1])
        loss_img = torch.tensor(0)

        if not self.latent_transformer:
            kld_loss = torch.tensor([0])
            loss = loss_rec + self.args.image_recon_lambda * loss_img
            elbo = loss_rec
            aux = 0
        else:
            # z_logit = z_logit.unsqueeze(1).repeat(1, output.size(1), 1)
            loss_aux = torch.tensor(0)  # self.criterion(
            # z_logit.reshape(-1, z_logit.size(-1)), target.reshape(-1))

            kl_weight = min(math.tanh(6 * self.kliter /
                                      self.args.full_kl_step - 3) + 1, 1)
            aux = loss_aux.item()
            elbo = loss_rec + kld_loss
            loss = loss_rec + self.args.kl_ceiling * kl_weight * kld_loss + \
                self.args.aux_ceiling*loss_aux + self.args.image_recon_lambda * loss_img

        return loss, loss_rec.item(), loss_img.item(), math.exp(min(loss_rec.item(), 100)), kld_loss.item(), aux, elbo.item()

    def training_step(self, batch, batch_idx):

        # switch to latent transformer if we've reached num_pretraining_steps
        if self.iter == self.args.num_pretraining_steps:
            self.latent_transformer = True
            self.model.switch_GVT_train_mode(self.latent_transformer)
            self.configure_optimizers()  # restart ADAM optimizer

        output, z_logit, kld_loss, image_recon = self(batch)
        target = batch["questions"].cuda()

        loss, loss_rec, loss_img, ppl, kld_loss, aux, elbo = self.calculate_losses(
            output, image_recon, kld_loss, z_logit, target)

        if self.latent_transformer:
            self.kliter += 1

        self.log('train loss', loss)
        self.log('train rec loss', loss_rec)
        self.log('image recon loss', loss_img)
        self.log('perplexity', ppl)
        self.log('kld loss', kld_loss)
        self.log('aux loss', aux)
        self.log('elbo', elbo)

        self.custom_optimizer(self.iter)
        self.iter += 1
        return loss

    def validation_step(self, batch, batch_idx):
        target = batch["questions"].cuda()
        output, z_logit, kld_loss, image_recon = self(batch)

        loss, loss_rec, loss_img, ppl, kld_loss, aux, elbo = self.calculate_losses(
            output, image_recon, kld_loss, z_logit, target)

        self.val_metrics["loss"].append(loss.item())
        self.val_metrics["img"].append(self.args.image_recon_lambda * loss_img)
        self.val_metrics["ppl"].append(ppl)
        self.val_metrics["kld"].append(kld_loss)
        self.val_metrics["aux"].append(aux)
        self.val_metrics["elbo"].append(elbo)
        self.val_metrics["rec"].append(loss_rec)

        self.log("val_loss", loss.item())
        self.log("val_loss_rec", loss_rec)
        self.log("val_img_loss", loss_img)
        self.log("val_ppl", ppl)
        self.log("val_kld_loss", kld_loss)
        self.log("val_aux", aux)
        self.log("val_elbo", elbo)

        return batch

    def validation_epoch_end(self, batch) -> None:

        print("##### End of Epoch validation #####")

        batch = batch[0]
        self.last_val_batch = batch

        print("VALIDATION SAMPLE")
        z_scores = self.decode_and_print(batch)

        for k, v in self.val_metrics.items():
            print(k, "\t", np.round(np.mean(v), 4))
            self.val_metrics[k] = []  # reset v

        for k, v in z_scores.items():
            rounded_val = np.round(np.mean(v) * 100, 4)
            self.log("val_"+k, rounded_val)
            print("z", k, "\t", rounded_val)

        print()
        print("This was validating after iteration {}".format(self.iter))

    def decode_and_print(self, batch, print_lim=20, testing=False):

        categories = batch["answer_types_for_input"].to(
            self.args.device)
        print_cats = batch["answer_types"].to(self.args.device).unsqueeze(1)
        if self.args.variant in ("lstm-c", "lstm-oc", "lstm-latentNorm-oc", "lstm-latent-oca-lt",
                                 "transformer-oc", "transformer-latentNorm-oc", "transformer-latent-oca-lt"):
            categories = batch["answer_type_orig"].to(self.args.device)

        images = batch["images"].to(self.args.device)
        rcnn_features = batch["rcnn_features"].to(self.args.device)
        rcnn_locations = batch["rcnn_locations"].to(self.args.device)
        image_ids = batch["image_ids"]
        questions = batch["questions"].to(self.args.device)

        z_preds = []
        gts = []
        z_decoded_sentences = self.model.decode_greedy(
            images, categories, questions, rcnn_features, rcnn_locations, max_decode_length=50)
        for i, greedy_sentence in enumerate(z_decoded_sentences):
            list_gt = self.filter_special_tokens(
                [self.vocab.idx2word[word] for word in batch["questions"][i].tolist()])
            z_list_pred = self.filter_special_tokens(greedy_sentence.split())
            gt = " ".join(list_gt)
            z_pred = " ".join(z_list_pred)
            gts.append(gt)
            z_preds.append(z_pred)
            if i < print_lim:
                print("Image ID:\t", image_ids[i])

                context_string = ""
                if self.args.variant == "lstm-c":
                    context_string = " ".join(
                        [self.cat2name[categories[i].item()]])
                else:
                    context_string = " ".join(
                        [self.vocab.idx2word[category] for category in print_cats[i].tolist()])
                print("Context:\t", context_string)
                print("z Generated: \t", z_pred)
                print("Reference: \t", gt)
                print()

        z_scores = self.nlge.compute_metrics(ref_list=[gts], hyp_list=z_preds)

        msd = MultisetDistances(references=gts)
        msj_distance = msd.get_jaccard_score(sentences=z_preds)
        new_msj_distance = {}
        for k in msj_distance.keys():
            new_msj_distance["msj_{}".format(k)] = msj_distance[k]
        z_scores.update(new_msj_distance)

        # # fbd = FBD(references=gts, model_name="bert-base-uncased", bert_model_dir="/homes/nv419/.cache/huggingface/transformers/")
        # # fbd_distance_sentences = fbd.get_score(sentences=z_preds)
        # # fbd = EMBD(references=gts, model_name="bert-base-uncased", bert_model_dir="/homes/nv419/.cache/huggingface/transformers/")
        # # embd_distance_sentences = fbd.get_score(sentences=z_preds)

        # z_scores.update({"fbd": fbd_distance_sentences, "embd": embd_distance_sentences})

        if not testing:
            for k, v in z_scores.items():
                rounded_val = np.round(np.mean(v) * 100, 4)
                if k == "Bleu_4":
                    self.bleus.append((self.iter, rounded_val))
                elif k == "msj_4":
                    self.msjs.append((self.iter, rounded_val))
                elif k == "fbd":
                    self.fbds.append((self.iter, rounded_val))

        max_bleu = max(self.bleus, key=itemgetter(1))
        max_msjs = max(self.msjs, key=itemgetter(1))
        # min_fbds = min(self.fbds, key=itemgetter(1))
        print("HIGHEST BLEU SCORE WAS: {} FROM ITER {}".format(
            max_bleu[1], max_bleu[0]))
        print("HIGHEST MSJ_4 SCORE WAS: {} FROM ITER {}".format(
            max_msjs[1], max_msjs[0]))
        # print("SMALLEST FBD SCORE WAS: {} FROM ITER {}".format(min_fbds[1], min_fbds[0]))

        print(self.hp_string)
        print(self.print_string)

        return z_scores

    def filter_special_tokens(self, decoded_sentence_list: List):

        if "<end>" in decoded_sentence_list:
            index_of_end = decoded_sentence_list.index("<end>")
            decoded_sentence_list = decoded_sentence_list[:index_of_end]

        filtered = []
        special_tokens = ["<start>", "<end>", "<pad>"]
        for token in decoded_sentence_list:
            if token not in special_tokens:
                filtered.append(token)
        return filtered

    def test_step(self, batch, batch_idx):
        images, questions, answers, categories = batch["images"], batch[
            "questions"], batch["answers"], batch["answer_types"]
        images, questions, answers, categories = images.to(self.args.device), questions.to(
            self.args.device), answers.to(self.args.device), categories.to(self.args.device)
        categories = categories.unsqueeze(1)

        z_scores = self.decode_and_print(batch, print_lim=10, testing=True)

        # for k, v in self.val_metrics.items():
        #     print(k, "\t", np.round(np.mean(v), 4))
        #     self.val_metrics[k] = []  # reset v

        for k, v in z_scores.items():
            print("z", k, "\t", np.round(np.mean(v) * 100, 4))

        for k, v in z_scores.items():
            if k not in self.test_scores.keys():
                self.test_scores[k] = []
            else:
                self.test_scores[k].append(v)

        return z_scores

    def test_end(self, all_scores):
        for k, scores in self.test_scores.items():
            self.test_scores[k] = np.mean(self.test_scores[k])

        print(self.test_scores)
        print(self.hp_string)
        return all_scores

    def custom_optimizer(self, step, warmup_steps=4000):
        pass
        # min_arg1 = m.sqrt(1/(step+1))
        # min_arg2 = step * (warmup_steps**-1.5)
        # lr = m.sqrt(1/self.args.hidden_dim) * min(min_arg1, min_arg2)

        # self.trainer.lightning_optimizers[0].param_groups[0]["lr"] = lr

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        optimizer = Lamb(self.parameters(), lr=args.lr,
                         weight_decay=0.01, betas=(.9, .999), adam=True)
        return optimizer


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224,
                                 scale=(1.00, 1.2),
                                 ratio=(0.75, 1.3333333333333333)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])


class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch=}_{global_step=}.ckpt"
            ckpt_path = os.path.join(
                trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)


class MyEarlyStopping(EarlyStopping):
    def on_validation_end(self, trainer, pl_module):
        if pl_module.iter > pl_module.args.num_pretraining_steps:
            self._run_early_stopping_check(trainer, pl_module)


early_stop_callback = MyEarlyStopping(
    monitor='val_Bleu_4',
    min_delta=0.00,
    patience=8,
    verbose=True,
    mode='max'
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model hyperparameters
    parser.add_argument("--emb_dim", type=int, default=300,
                        help="Embedding dimensionality of the model")
    parser.add_argument("--hidden_dim", type=int, default=300,
                        help="Hidden dimensionality of the model")
    parser.add_argument("--latent_dim", type=int, default=300,
                        help="Size of latent dimension")
    parser.add_argument("--pwffn_dim", type=int, default=600,
                        help="Size of postionwise feedforward network in transformer")
    parser.add_argument("--num_layers", type=int, default=4,
                        help="Number of transformer layers in encoder and decoder")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Number of heads in the multi-head attention")
    parser.add_argument("--lr", type=float, default=3e-5,
                        help="Learning rate of the network")
    parser.add_argument("--num_pretraining_steps", type=float, default=12000,
                        help="Number of pretraining steps before turning on latent transformer")
    parser.add_argument("--total_training_steps", type=int, default=35000,
                        help="Total number of training steps for the model")
    parser.add_argument("--full_kl_step", type=int, default=15000,
                        help="Number of steps until KLD is annealed")
    parser.add_argument("--kl_ceiling", type=float, default=0.5)
    parser.add_argument("--aux_ceiling", type=float, default=1.0)
    parser.add_argument("--image_recon_lambda", type=float, default=0.1,
                        help="How much to scale the image reconstruction loss by")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--early_stop", type=lambda x: bool(strtobool(x)), default=True)
    # Data args
    parser.add_argument("--emb_file", type=str, default="vectors/glove.6B.300d.txt",
                        help="Filepath for pretrained embeddings")
    parser.add_argument("--dataset", type=str,
                        default="data/processed/iq_dataset.hdf5")
    parser.add_argument("--val_dataset", type=str,
                        default="data/processed/iq_val_dataset.hdf5")
    parser.add_argument("--vocab", type=str, default="vocab.pkl")
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--print_note", type=str, default="")
    parser.add_argument("--input_mode", type=str, default="ans")
    parser.add_argument("--variant", type=str, default="lstm-baseline",
                        help="o = object, c = category, a = answer (train only), q = question (train only), d = caption (train only)")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available()
                          and args.use_gpu else 'cpu')
    args.device = device
    args.root_dir = os.getcwd()

    if os.path.exists(args.vocab):
        vocab = pickle.load(open(args.vocab, "rb"))
    else:
        vocab = build_vocab(
            'data/vqa/v2_OpenEnded_mscoco_train2014_questions.json', 'data/vqa/iq_dataset.json', 4)

    data_loader = get_loader(os.path.join(
        os.getcwd(), args.dataset), transform, args.batch_size, shuffle=True, num_workers=8)
    val_data_loader = get_loader(os.path.join(
        os.getcwd(), args.val_dataset), transform, args.batch_size, shuffle=True, num_workers=8)

    trainGVT = TrainIQ(vocab, args).to(args.device)
    if args.early_stop:
        trainer = pl.Trainer(max_steps=args.total_training_steps, gradient_clip_val=5,
                             val_check_interval=250, limit_val_batches=135, gpus=args.num_gpus, callbacks=[early_stop_callback])
    else:
        trainer = pl.Trainer(max_steps=args.total_training_steps, gradient_clip_val=5,
                             val_check_interval=250, limit_val_batches=135, gpus=args.num_gpus)

    trainer.fit(trainGVT, data_loader, val_data_loader)

    test_data_loader = get_loader(os.path.join(
        os.getcwd(), args.val_dataset), transform, args.batch_size, shuffle=False, num_workers=8)
    trainer.test(trainGVT, test_dataloaders=test_data_loader, ckpt_path="best")

    for k, scores in trainGVT.test_scores.items():
        print(k, np.mean(scores))
