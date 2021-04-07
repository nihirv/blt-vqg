"""Contains code for the IQ model.
"""

from math import log
from models.deep_questioner import DeepQuestioner
import os
# from utils import vocab
from models import transformer_layers
from models.decoder_transformer import GVTransformerDecoder
from models.transformer_layers import Decoder, Latent, generate_pad_mask
from models.encoder_transformer import GVTransformerEncoder
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder_cnn import EncoderCNN
from .mlp import MLP


class IQ(nn.Module):
    """Information Maximization question generation.
    """
    def __init__(self, latent_transformer, vocab, args, num_att_layers=2):
        super(IQ, self).__init__()
        self.vocab = vocab
        self.vocab_size = len(self.vocab.word2idx)
        self.latent_transformer = latent_transformer
        self.args = args

        self.embedding = self.embedder()


        # Setup image encoder.
        self.encoder_cnn = EncoderCNN(args)

        self.latent_layer = Latent(args)
        self.latent_projection = nn.Linear(args.latent_dim, args.hidden_dim)

        self.answer_encoder = GVTransformerEncoder(self.embedding, self.latent_layer, self.latent_transformer, args)
        self.deep_questioner = DeepQuestioner(args)
        self.decoder = GVTransformerDecoder(self.embedding, self.latent_transformer, self.vocab_size, vocab, args)

        # Setup image reconstruction.
        self.image_reconstructor = MLP(
                args.hidden_dim, args.pwffn_dim, args.hidden_dim,
                num_layers=num_att_layers)


    def switch_GVT_train_mode(self, new_mode):
        self.latent_transformer = new_mode
        self.answer_encoder.latent_transformer = new_mode
        self.decoder.latent_transformer = new_mode


    def embedder(self):
        init_embeddings = np.random.randn(self.vocab_size, self.args.emb_dim) * 0.01 
        print('Embeddings: %d x %d' % (self.vocab_size, self.args.emb_dim))
        if self.args.emb_file is not None:
            print('Loading embedding file: %s' % os.path.join(self.args.root_dir, self.args.emb_file))
            pre_trained = 0
            for line in open(os.path.join(self.args.root_dir, self.args.emb_file)).readlines():
                sp = line.split()
                if(len(sp) == self.args.emb_dim + 1):
                    if sp[0] in self.vocab.word2idx:
                        pre_trained += 1
                        init_embeddings[self.vocab.word2idx[sp[0]]] = [float(x) for x in sp[1:]]
                else:
                    print(sp[0])
            print('Pre-trained: %d (%.2f%%)' % (pre_trained, pre_trained * 100.0 / self.vocab_size))
        embedding = nn.Embedding(self.vocab_size, self.args.emb_dim, padding_idx=self.vocab.word2idx[self.vocab.SYM_PAD])
        embedding.weight.data.copy_(torch.FloatTensor(init_embeddings))
        embedding.weight.data.requires_grad = True
        emb_projection = nn.Sequential(
            embedding,
            nn.Linear(self.args.emb_dim, self.args.hidden_dim)
        )
        return emb_projection


    def forward(self, images, answers, response, target):
        """Passes the image and the question through a model and generates answers.

        Args:
            images: Batch of image Variables.
            answers: Batch of answer Variables.
            categories: Batch of answer Variables.
            alengths: List of answer lengths.
            questions: Batch of question Variables.
            teacher_forcing_ratio: Whether to predict with teacher forcing.
            decode_function: What to use when choosing a word from the
                distribution over the vocabulary.

        Returns:
            - outputs: The output scores for all steps in the RNN.
            - hidden: The hidden states of all the RNNs.
            - ret_dict: A dictionary of attributes. See DecoderRNN.py for details.
        """
        # features is (N * args.hidden_dim)
        image_features, feature_maps = self.encoder_cnn(images)

        # z-path. transformer_posteriors is a tuple: (mean_posterior, logvar_posterior)
        encoder_outputs, response_encoder_outputs, src_mask = self.answer_encoder(answers, response, image_features)
        # deep_questioner_outputs = self.deep_questioner(encoder_outputs, feature_maps, src_mask)
        # deep_questioner_outputs = self.deep_questioner(encoder_outputs, feature_maps, src_mask)
        # deep_questioner_outputs = torch.cat((encoder_outputs, deep_questioner_outputs), dim=1)
        # falses_mask = torch.zeros(images.shape[0], 1, deep_questioner_outputs.shape[1] - src_mask.shape[-1]).to(self.args.device)
        # src_mask = torch.cat((falses_mask, src_mask), dim=-1)

        
        kld_loss, z, posteriors = None, None, None
        if self.latent_transformer:
            kld_loss, z, posteriors = self.latent_layer(encoder_outputs[:, 0], response_encoder_outputs[:, 0])
            z = self.latent_projection(z)
        
        output, z_logit = self.decoder(encoder_outputs, target, image_features, feature_maps, z, src_mask)

        if self.latent_transformer: # experiement without requiring the latent mode enabled?
            reconstructed_image_features = self.image_reconstructor(encoder_outputs[:, 0] + z)
        else:
            reconstructed_image_features = self.image_reconstructor(encoder_outputs[:, 0])

        return output, z_logit, kld_loss, (image_features, reconstructed_image_features)


    def decode_greedy(self, images, answers, max_decode_length = 50):
        image_features, feature_maps = self.encoder_cnn(images)
        src_mask = generate_pad_mask(answers)
        embedded_context = self.embedding(answers)
        encoder_outputs = self.answer_encoder.encoder(embedded_context, src_mask)
        # deep_questioner_outputs = self.deep_questioner(encoder_outputs, feature_maps, src_mask)
        # deep_questioner_outputs = torch.cat((encoder_outputs, deep_questioner_outputs), dim=1)
        # falses_mask = torch.zeros(images.shape[0], 1, deep_questioner_outputs.shape[1] - src_mask.shape[-1]).to(self.args.device)
        # src_mask = torch.cat((falses_mask, src_mask), dim=-1)

        z = 0
        if self.latent_transformer:
            _, z, _ = self.latent_layer(encoder_outputs[:, 0], None)
            z = self.latent_projection(z)

        ys = torch.ones(answers.shape[0], 1).fill_(self.vocab.word2idx[self.vocab.SYM_PAD]).long().to(self.args.device)
        top_args = torch.zeros(answers.shape[0], max_decode_length+1, 6).to(self.args.device)
        top_args_vals = torch.zeros(answers.shape[0], max_decode_length+1, 6).to(self.args.device)

        decoded_words = []
        for i in range(max_decode_length + 1):
            pred_targets_logit = self.decoder.inference_forward(encoder_outputs, ys, image_features, feature_maps, z, src_mask)
            _, pred_next_word = torch.max(pred_targets_logit[:, -1], dim=1)
            top_6_vals, top_6_indicies = torch.topk(torch.nn.functional.softmax(pred_targets_logit[:, -1], -1), 6, dim=1)

            decoded_words.append(['<end>' if token.item() == self.vocab.word2idx[self.vocab.SYM_EOS] else self.vocab.idx2word[token.item()] for token in pred_next_word.view(-1)])

            ys = torch.cat([ys, pred_next_word.unsqueeze(1)], dim=1)
            top_args[:, i] = top_6_indicies
            top_args_vals[:, i] = top_6_vals

        sentence = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ''
            for e in row:
                if e == '<end>': break
                else: st+= e + ' '
            sentence.append(st)
        return sentence, top_args, top_args_vals
