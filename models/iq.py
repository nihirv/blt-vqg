"""Contains code for the IQ model.
"""

from math import log
import os
# from utils import vocab
from models import transformer_layers
from models.decoder_transformer import GVTransformerDecoder
from models.transformer_layers import Encoder, Latent, LatentTwoSpaces, gaussian_kld, gaussian_kld_unit_norm, generate_pad_mask
from models.encoder_transformer import GVTransformerEncoder
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder_cnn import EncoderCNN
from .encoder_rnn import EncoderRNN
from .decoder_rnn import DecoderRNN
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

        # z-path
        self.latent_layer = LatentTwoSpaces(args, args.dropout)
        self.latent_projection = nn.Linear(args.latent_dim, args.hidden_dim)
        self.answer_encoder = GVTransformerEncoder(self.embedding, self.latent_layer, self.latent_transformer, args)
        self.z_classifier = nn.Linear(args.hidden_dim, self.vocab_size)

        # t-path
        self.category_embedding, self.category_image_encoder, self.t_latent_layer, self.t_latent_projection, self.t_classifier = None, None, None, None, None
        if args.enable_t_space:
            self.category_embedding = nn.Embedding(16, args.hidden_dim) # maybe tinker with this hidden size? Seems very big
            self.category_image_encoder = nn.Sequential(
                nn.Linear(args.hidden_dim, args.pwffn_dim),
                nn.Dropout(0.3),
                nn.ELU(),
                nn.Linear(args.pwffn_dim, args.hidden_dim)
            )

            self.category_image_encoder_projection = nn.Linear(args.hidden_dim, args.hidden_dim)
        
            self.t_latent_layer = Latent(args)
            self.t_latent_projection = nn.Linear(args.latent_dim, args.hidden_dim)
            self.t_classifier = nn.Linear(args.hidden_dim, self.vocab_size)


        self.decoder = GVTransformerDecoder(self.embedding, self.z_classifier, self.t_classifier, self.latent_transformer, self.vocab_size, vocab, args)
        self.l2Loss = nn.MSELoss()

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

    def encode_category_images(self, categories, image_features):
        category_embedding = self.category_embedding(categories)
        cat_image_features = (category_embedding + image_features)#.unsqueeze(1)
        encoded_category_image = self.category_image_encoder(cat_image_features)#, mask=None)
        encoded_category_image = self.category_image_encoder_projection(encoded_category_image)
        # encoded_category_image = encoded_category_image
        return encoded_category_image




    def forward(self, images, answers, response, target, categories):

        # features is (N * args.hidden_dim)
        image_features = self.encoder_cnn(images)

        # z-path. transformer_priors/posteriors is a tuple: (mean_prior, logvar_prior)
        encoder_outputs, response_outputs, z_kld, z, transformer_priors, transformer_posteriors, src_mask, res_mask = self.answer_encoder(answers, response, image_features)

        # t-path.
        l2_category_encoder = torch.tensor([0]).to(self.args.device).detach()
        if self.args.enable_t_space:
            encoded_category_image = self.encode_category_images(categories, image_features)
            # l2_category_encoder = self.l2Loss(encoded_category_image, encoder_outputs[:,0])

        z_t_kld, t_kld, t = torch.tensor([0]).to(self.args.device), torch.tensor([0]).to(self.args.device), None
        if self.latent_transformer:
            z = self.latent_projection(z)

            if self.args.enable_t_space:
                # during latent training, update t with z posteriors
                # t_kld, t, t_priors, t_posteriors = self.t_latent_layer(encoded_category_image, response_outputs[:, 0])
                t_kld, t, t_priors, t_posteriors = self.t_latent_layer(encoded_category_image)
                t = self.t_latent_projection(t)
                z_t_kld = torch.mean(gaussian_kld(*t_posteriors, *transformer_posteriors))
                # z_t_kld = torch.mean(gaussian_kld(*transformer_priors, *t_priors))



        output, z_logit, t_logit = self.decoder(encoder_outputs, target, image_features, z, t, src_mask)

        response_output = None
        if self.args.reconstruct_through_response:
            response_output, _, _ = self.decoder(response_outputs, target, image_features, z, t, res_mask)

        if self.latent_transformer: # experiement without requiring the latent mode enabled?
            reconstructed_image_features = self.image_reconstructor(encoder_outputs[:, 0] + z)
        else:
            reconstructed_image_features = self.image_reconstructor(encoder_outputs[:, 0])

        return_logits = (z_logit, t_logit)
        return_klds = (z_kld, t_kld, z_t_kld)
        return_image_features = (image_features, reconstructed_image_features)
        return_transformer_latents = (transformer_priors, transformer_posteriors)

        return output, response_output, return_logits, l2_category_encoder, return_klds, return_image_features, return_transformer_latents


    def decode_greedy(self, images, categories, max_decode_length = 50):
        image_features = self.encoder_cnn(images)
        src_mask = generate_pad_mask(categories)
        embedded_context = self.embedding(categories)
        ###
        # one_tensor = torch.LongTensor([0]).to(self.args.device).detach()
        # category_encoding_vector = self.answer_encoder.category_segment_encoding(one_tensor)
        # embedded_context[:, :2] = embedded_context[:, :2] + category_encoding_vector
        ###
        encoder_outputs = self.answer_encoder.encoder(embedded_context, src_mask)
        encoder_outputs[:, 0] = encoder_outputs[:, 0] + image_features # TEST THISSS
        if self.args.enable_t_space:
            categories = categories.squeeze(1)
            t_encoder_outputs = self.encode_category_images(categories, image_features)


        z_latent, t_latent = 0, 0
        if self.latent_transformer:
            if self.args.enable_t_space:
                _, latent, _, _ = self.t_latent_layer(t_encoder_outputs, None)
                t_latent = self.t_latent_projection(latent)
            else:
                _, latent, _, _ = self.latent_layer(encoder_outputs[:, 0], None)
                z_latent = self.latent_projection(latent)

        z_ys = torch.ones(categories.shape[0], 1).fill_(self.vocab.word2idx[self.vocab.SYM_PAD]).long().to(self.args.device)
        t_ys = torch.ones(categories.shape[0], 1).fill_(self.vocab.word2idx[self.vocab.SYM_PAD]).long().to(self.args.device)
        # top_args = torch.zeros(categories.shape[0], max_decode_length+1, 6).to(self.args.device)
        # top_args_vals = torch.zeros(categories.shape[0], max_decode_length+1, 6).to(self.args.device)

        z_decoded_words = []
        t_decoded_words = []
        for i in range(max_decode_length + 1):
            pred_targets_logit = self.decoder.inference_forward(encoder_outputs, z_ys, image_features, z_latent, src_mask)
            _, pred_next_word = torch.max(pred_targets_logit[:, -1], dim=1)
            z_decoded_words.append(['<end>' if token.item() == self.vocab.word2idx[self.vocab.SYM_EOS] else self.vocab.idx2word[token.item()] for token in pred_next_word.view(-1)])
            z_ys = torch.cat([z_ys, pred_next_word.unsqueeze(1)], dim=1)

            if self.args.enable_t_space:
                pred_targets_logit = self.decoder.inference_forward(t_encoder_outputs.unsqueeze(1), t_ys, image_features, t_latent, src_mask)
                _, pred_next_word = torch.max(pred_targets_logit[:, -1], dim=1)
                t_decoded_words.append(['<end>' if token.item() == self.vocab.word2idx[self.vocab.SYM_EOS] else self.vocab.idx2word[token.item()] for token in pred_next_word.view(-1)])
                t_ys = torch.cat([t_ys, pred_next_word.unsqueeze(1)], dim=1)
            # top_6_vals, top_6_indicies = torch.topk(torch.nn.functional.softmax(pred_targets_logit[:, -1], -1), 6, dim=1)
            # top_args[:, i] = top_6_indicies
            # top_args_vals[:, i] = top_6_vals

        z_sentence = self.process_to_list(z_decoded_words)
        t_sentence = self.process_to_list(t_decoded_words)

        return z_sentence, t_sentence

    def process_to_list(self, decoded_words):
        sentence = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ''
            for e in row:
                if e == '<end>': break
                else: st+= e + ' '
            sentence.append(st)
        return sentence