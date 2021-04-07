"""Contains code for the IQ model.
"""

from math import log
import math
from models.new_transformer_encoder import TransformerModel
import os
import random
from numpy.lib.type_check import imag
from torch.functional import Tensor

from torch.nn.modules.loss import CrossEntropyLoss
# from utils import vocab
from models import transformer_layers
from models.decoder_transformer import GVTransformerDecoder
from models.transformer_layers import Decoder, ImageTransformerEncoder, Latent, LatentNorm, generate_pad_mask
from models.encoder_transformer import GVTransformerEncoder
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder_cnn import EncoderCNN
from .decoder_lstm import Attention, Decoder
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

        self.encoder_cnn = EncoderCNN(args)
        self.image_reconstructor = MLP(
            args.hidden_dim, args.pwffn_dim, args.hidden_dim,
            num_layers=num_att_layers)

        self.category_embedding = None
        self.category_image_proj = None
        self.latent_layer = None
        self.latent_projection = None
        self.text_encoder_T = None
        self.image_encoder_T = None
        self.decoder_T = None
        self.decoder_attention_R = None
        self.decoder_R = None
        self.image_reconstructor = None

        if self.args.variant == "lstm-baseline":
            self.decoder_attention_R = Attention(args.hidden_dim)
            self.decoder_R = Decoder(
                args, self.vocab_size, self.decoder_attention_R)

        if self.args.variant == "transformer-baseline":
            self.embedding = self.embedder()
            self.decoder_T = GVTransformerDecoder(
                self.embedding, self.latent_transformer, self.vocab_size, vocab, args)

        if self.args.variant == "lstm-o":
            self.decoder_attention_R = Attention(args.hidden_dim)
            self.decoder_R = Decoder(
                args, self.vocab_size, self.decoder_attention_R)
            self.image_encoder_T = ImageTransformerEncoder(args)

        if self.args.variant == "transformer-o":
            self.embedding = self.embedder()
            self.decoder_T = GVTransformerDecoder(
                self.embedding, self.latent_transformer, self.vocab_size, vocab, args)
            self.image_encoder_T = ImageTransformerEncoder(args)

        if self.args.variant == "lstm-c":
            self.category_embedding = nn.Embedding(16, args.emb_dim)
            self.category_image_proj = nn.Linear(
                args.hidden_dim + args.emb_dim, args.hidden_dim)

            self.decoder_attention_R = Attention(args.hidden_dim)
            self.decoder_R = Decoder(
                args, self.vocab_size, self.decoder_attention_R)

        if self.args.variant == "transformer-c":
            self.embedding = self.embedder()
            self.text_encoder = TransformerModel(
                args.hidden_dim, args.num_heads, args.pwffn_dim, args.num_layers, args.dropout, args)
            self.decoder_T = GVTransformerDecoder(
                self.embedding, self.latent_transformer, self.vocab_size, vocab, args)

        if self.args.variant == "lstm-oc":
            self.category_embedding = nn.Embedding(16, args.emb_dim)
            self.category_image_proj = nn.Linear(
                args.hidden_dim + args.emb_dim, args.hidden_dim)
            self.image_encoder_T = ImageTransformerEncoder(args)
            self.decoder_attention_R = Attention(args.hidden_dim)
            self.decoder_R = Decoder(
                args, self.vocab_size, self.decoder_attention_R)

        if self.args.variant == "transformer-oc":
            self.category_embedding = nn.Embedding(16, args.emb_dim)
            self.category_image_proj = nn.Linear(
                args.hidden_dim + args.emb_dim, args.hidden_dim)
            self.image_encoder_T = ImageTransformerEncoder(args)
            self.embedding = self.embedder()
            self.decoder_T = GVTransformerDecoder(
                self.embedding, self.latent_transformer, self.vocab_size, vocab, args)

        if self.args.variant == "lstm-ca":
            self.embedding = self.embedder()
            self.text_encoder = TransformerModel(
                args.hidden_dim, args.num_heads, args.pwffn_dim, args.num_layers, args.dropout, args)
            self.decoder_attention_R = Attention(args.hidden_dim)
            self.decoder_R = Decoder(
                args, self.vocab_size, self.decoder_attention_R)

        if self.args.variant == "transformer-ca":
            self.embedding = self.embedder()
            self.text_encoder = TransformerModel(
                args.hidden_dim, args.num_heads, args.pwffn_dim, args.num_layers, args.dropout, args)
            self.decoder_T = GVTransformerDecoder(
                self.embedding, self.latent_transformer, self.vocab_size, vocab, args)

        if self.args.variant == "lstm-oca":
            self.embedding = self.embedder()
            self.image_encoder_T = ImageTransformerEncoder(args)
            self.text_encoder = TransformerModel(
                args.hidden_dim, args.num_heads, args.pwffn_dim, args.num_layers, args.dropout, args)
            self.decoder_attention_R = Attention(args.hidden_dim)
            self.decoder_R = Decoder(
                args, self.vocab_size, self.decoder_attention_R)

        if self.args.variant == "transformer-oca":
            self.embedding = self.embedder()
            self.image_encoder_T = ImageTransformerEncoder(args)
            self.text_encoder = TransformerModel(
                args.hidden_dim, args.num_heads, args.pwffn_dim, args.num_layers, args.dropout, args)
            self.decoder_T = GVTransformerDecoder(
                self.embedding, self.latent_transformer, self.vocab_size, vocab, args)

        if self.args.variant == "lstm-oca-lt":
            self.embedding = self.embedder(project=False)
            self.text_embedder = nn.LSTM(
                input_size=args.emb_dim, hidden_size=args.hidden_dim, batch_first=True)
            self.image_encoder_T = ImageTransformerEncoder(args)
            self.text_encoder = TransformerModel(
                args.hidden_dim, args.num_heads, args.pwffn_dim, args.num_layers, args.dropout, args)
            self.decoder_attention_R = Attention(args.hidden_dim)
            self.decoder_R = Decoder(
                args, self.vocab_size, self.decoder_attention_R)

        if self.args.variant == "transformer-oca-lt":
            self.embedding = self.embedder()
            self.text_embedder = nn.LSTM(
                input_size=args.hidden_dim, hidden_size=args.hidden_dim, batch_first=True)
            self.image_encoder_T = ImageTransformerEncoder(args)
            self.text_encoder = TransformerModel(
                args.hidden_dim, args.num_heads, args.pwffn_dim, args.num_layers, args.dropout, args)
            self.decoder_T = GVTransformerDecoder(
                self.embedding, self.latent_transformer, self.vocab_size, vocab, args)

        if self.args.variant == "lstm-latentNorm-oc":
            self.category_embedding = nn.Embedding(16, args.emb_dim)
            self.category_image_proj = nn.Linear(
                args.hidden_dim + args.emb_dim, args.hidden_dim)
            self.image_encoder_T = ImageTransformerEncoder(args)
            self.latent_layer = LatentNorm(args)
            self.decoder_attention_R = Attention(args.hidden_dim)
            self.decoder_R = Decoder(
                args, self.vocab_size, self.decoder_attention_R)

        if self.args.variant == "lstm-latent-oca-lt":
            self.category_embedding = nn.Embedding(16, args.emb_dim)
            self.category_image_proj = nn.Linear(
                args.hidden_dim + args.emb_dim, args.hidden_dim)
            self.embedding = self.embedder(project=False)
            self.text_embedder = nn.LSTM(
                input_size=args.emb_dim, hidden_size=args.hidden_dim, batch_first=True)
            self.text_encoder = TransformerModel(
                args.hidden_dim, args.num_heads, args.pwffn_dim, args.num_layers, args.dropout, args)
            self.image_encoder_T = ImageTransformerEncoder(args)
            self.latent_layer = Latent(args)
            self.decoder_attention_R = Attention(args.hidden_dim)
            self.decoder_R = Decoder(
                args, self.vocab_size, self.decoder_attention_R)

        if self.args.variant == "transformer-latentNorm-oc":
            self.category_embedding = nn.Embedding(16, args.emb_dim)
            self.category_image_proj = nn.Linear(
                args.hidden_dim + args.emb_dim, args.hidden_dim)
            self.image_encoder_T = ImageTransformerEncoder(args)
            self.latent_layer = LatentNorm(args)
            self.embedding = self.embedder()
            self.decoder_T = GVTransformerDecoder(
                self.embedding, self.latent_transformer, self.vocab_size, vocab, args)

        if self.args.variant == "transformer-latent-oca-lt":
            self.category_embedding = nn.Embedding(16, args.emb_dim)
            self.category_image_proj = nn.Linear(
                args.hidden_dim + args.emb_dim, args.hidden_dim)
            self.embedding = self.embedder()
            self.text_embedder = nn.LSTM(
                input_size=args.hidden_dim, hidden_size=args.hidden_dim, batch_first=True)
            self.text_encoder = TransformerModel(
                args.hidden_dim, args.num_heads, args.pwffn_dim, args.num_layers, args.dropout, args)
            self.image_encoder_T = ImageTransformerEncoder(args)
            self.latent_layer = Latent(args)
            self.embedding = self.embedder()
            self.decoder_T = GVTransformerDecoder(
                self.embedding, self.latent_transformer, self.vocab_size, vocab, args)

            # Setup image encoder.
            # self.image_encoder = ImageTransformerEncoder(args)

            # self.category_embedding = nn.Embedding(16, args.emb_dim)
            # self.category_image_proj = nn.Linear(
            #     args.hidden_dim + args.emb_dim, args.hidden_dim)

            # self.latent_layer = Latent(args)
            # self.latent_projection = nn.Linear(args.latent_dim, args.hidden_dim)

            # self.answer_encoder = GVTransformerEncoder(
            #     self.embedding, None, self.latent_transformer, args)  # TODO: REMEMBER TO ENABLE LATENT AGAIN

            # self.decoder = GVTransformerDecoder(
            #     self.embedding, self.latent_transformer, self.vocab_size, vocab, args)

            # Setup image reconstruction.

        self.ce = CrossEntropyLoss(
            ignore_index=self.vocab.word2idx[self.vocab.SYM_PAD])

    def switch_GVT_train_mode(self, new_mode):
        self.latent_transformer = new_mode
        try:
            self.decoder_T.latent_transformer = new_mode
        except:
            pass

    def embedder(self, project=True):
        init_embeddings = np.random.randn(
            self.vocab_size, self.args.emb_dim) * 0.01
        print('Embeddings: %d x %d' % (self.vocab_size, self.args.emb_dim))
        if self.args.emb_file is not None:
            print('Loading embedding file: %s' %
                  os.path.join(self.args.root_dir, self.args.emb_file))
            pre_trained = 0
            for line in open(os.path.join(self.args.root_dir, self.args.emb_file)).readlines():
                sp = line.split()
                if(len(sp) == self.args.emb_dim + 1):
                    if sp[0] in self.vocab.word2idx:
                        pre_trained += 1
                        init_embeddings[self.vocab.word2idx[sp[0]]] = [
                            float(x) for x in sp[1:]]
                else:
                    print(sp[0])
            print('Pre-trained: %d (%.2f%%)' %
                  (pre_trained, pre_trained * 100.0 / self.vocab_size))
        embedding = nn.Embedding(self.vocab_size, self.args.emb_dim,
                                 padding_idx=self.vocab.word2idx[self.vocab.SYM_PAD])
        embedding.weight.data.copy_(torch.FloatTensor(init_embeddings))
        embedding.weight.data.requires_grad = True
        if project:
            embedding = nn.Sequential(
                embedding,
                nn.Linear(self.args.emb_dim, self.args.hidden_dim)
            )
        return embedding

    def decode_rnn(self, encoder_outputs, z, target, teacher_forcing_ratio=0.5):
        outputs = torch.zeros(
            target.shape[1], target.shape[0], self.vocab_size, device=self.args.device)
        input = target[:, 0]
        hidden = z.contiguous()

        for t in range(1, target.shape[1]):
            output, hidden = self.decoder_R(
                input, hidden, encoder_outputs.contiguous())
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = target[:, t] if teacher_force else top1

        return outputs.permute(1, 0, 2)  # [B, T, D]

    def encode_object_features(self, rcnn_features, rcnn_locations):
        encoded_objects, image_pad_mask = self.image_encoder_T(
            rcnn_features, rcnn_locations)  # encoded_objects [T, B, D]
        encoded_objects = encoded_objects.permute(1, 0, 2)  # [B, T, D]
        return encoded_objects, image_pad_mask.unsqueeze(1)

    def forward(self, images, category, answers, response, target, rcnn_features, rcnn_locations, teacher_forcing=0.5):
        # features is (N * args.hidden_dim)
        image_features = self.encoder_cnn(images)
        kld_loss = torch.tensor(0)

        z_logit = None
        if self.args.variant == "lstm-baseline":
            decoder_outputs = self.decode_rnn(
                image_features.unsqueeze(1), image_features, target)  # [B, T, D]

        if self.args.variant == "transformer-baseline":
            image_pad_mask = torch.zeros(
                images.shape[0], 1, 1, device=self.args.device).bool()
            decoder_outputs, z_logit = self.decoder_T(
                image_features.unsqueeze(1), target, image_features, image_features, image_pad_mask)

        if self.args.variant == "lstm-o":
            encoded_objects, _ = self.encode_object_features(
                rcnn_features, rcnn_locations)
            decoder_outputs = self.decode_rnn(
                encoded_objects, image_features, target)  # [B, T, D]

        if self.args.variant == "transformer-o":
            encoded_objects, image_pad_mask = self.encode_object_features(
                rcnn_features, rcnn_locations)
            decoder_outputs, z_logit = self.decoder_T(
                encoded_objects, target, image_features, image_features, image_pad_mask)

        if self.args.variant == "lstm-c":
            category_features = self.category_embedding(category)
            cat_features = torch.cat((category_features, image_features), -1)
            cat_img = self.category_image_proj(cat_features)
            decoder_outputs = self.decode_rnn(
                cat_img.unsqueeze(1), cat_img, target)  # [B, T, D]

        if self.args.variant == "transformer-c":
            src_mask = self.text_encoder.generate_square_subsequent_mask(
                answers.shape[-1]).to(self.args.device)
            embedded_context = self.embedding(
                answers) * math.sqrt(self.args.hidden_dim)
            encoder_outputs = self.text_encoder(embedded_context.permute(
                1, 0, 2), src_mask).permute(1, 0, 2)  # [B, T, D]
            decoder_outputs, z_logit = self.decoder_T(
                encoder_outputs, target, image_features, encoder_outputs[:, 1], generate_pad_mask(
                    answers)
            )

        if self.args.variant == "lstm-oc":
            category_features = self.category_embedding(category)
            cat_features = torch.cat((category_features, image_features), -1)
            cat_img = self.category_image_proj(cat_features)
            encoded_objects, _ = self.encode_object_features(
                rcnn_features, rcnn_locations)
            decoder_outputs = self.decode_rnn(
                encoded_objects, cat_img, target)  # [B, T, D]

        if self.args.variant == "transformer-oc":
            category_features = self.category_embedding(category)
            cat_features = torch.cat((category_features, image_features), -1)
            cat_img = self.category_image_proj(cat_features)
            encoded_objects, image_pad_mask = self.encode_object_features(
                rcnn_features, rcnn_locations)
            decoder_outputs, z_logit = self.decoder_T(
                encoded_objects, target, cat_img, cat_img, image_pad_mask
            )

        if self.args.variant == "lstm-ca":
            src_mask = self.text_encoder.generate_square_subsequent_mask(
                answers.shape[-1]).to(self.args.device)
            embedded_context = self.embedding(
                answers) * math.sqrt(self.args.hidden_dim)
            encoder_outputs = self.text_encoder(embedded_context.permute(
                1, 0, 2), src_mask).permute(1, 0, 2)  # [B, T, D]
            encoder_outputs[:, 1] = encoder_outputs[:, 1] + image_features
            decoder_outputs = self.decode_rnn(
                encoder_outputs, encoder_outputs[:, 1], target
            )

        if self.args.variant == "transformer-ca":
            src_mask = self.text_encoder.generate_square_subsequent_mask(
                answers.shape[-1]).to(self.args.device)
            embedded_context = self.embedding(
                answers) * math.sqrt(self.args.hidden_dim)
            encoder_outputs = self.text_encoder(embedded_context.permute(
                1, 0, 2), src_mask).permute(1, 0, 2)  # [B, T, D]
            decoder_outputs, z_logit = self.decoder_T(
                encoder_outputs, target, image_features, encoder_outputs[:, 1], generate_pad_mask(
                    answers)
            )

        if self.args.variant == "lstm-oca":
            encoded_objects, _ = self.encode_object_features(
                rcnn_features, rcnn_locations)
            src_mask = self.text_encoder.generate_square_subsequent_mask(
                answers.shape[-1]).to(self.args.device)
            embedded_context = self.embedding(
                answers) * math.sqrt(self.args.hidden_dim)
            encoder_outputs = self.text_encoder(embedded_context.permute(
                1, 0, 2), src_mask).permute(1, 0, 2)  # [B, T, D]
            encoder_outputs[:, 1] = encoder_outputs[:, 1] + image_features
            encoder_outputs = torch.cat(
                (encoded_objects, encoder_outputs), dim=1)
            decoder_outputs = self.decode_rnn(
                encoder_outputs, encoder_outputs[:, 1], target
            )

        if self.args.variant == "transformer-oca":
            encoded_objects, _ = self.encode_object_features(
                rcnn_features, rcnn_locations)
            src_mask = self.text_encoder.generate_square_subsequent_mask(
                answers.shape[-1]).to(self.args.device)
            embedded_context = self.embedding(
                answers) * math.sqrt(self.args.hidden_dim)
            encoder_outputs = self.text_encoder(embedded_context.permute(
                1, 0, 2), src_mask).permute(1, 0, 2)  # [B, T, D]
            encoder_outputs[:, 1] = encoder_outputs[:, 1] + image_features
            encoder_outputs = torch.cat(
                (encoded_objects, encoder_outputs), dim=1)
            image_mask = torch.zeros(
                encoded_objects.shape[0], 1, 36, device=self.args.device).bool()
            src_mask = torch.cat(
                (image_mask, generate_pad_mask(answers)), dim=-1)
            decoder_outputs, z_logit = self.decoder_T(
                encoder_outputs, target, image_features, encoder_outputs[:, 1], src_mask
            )

        if self.args.variant == "lstm-oca-lt":
            encoded_objects, _ = self.encode_object_features(
                rcnn_features, rcnn_locations)
            src_mask = self.text_encoder.generate_square_subsequent_mask(
                answers.shape[-1]).to(self.args.device)
            embedded_context = self.embedding(
                answers) * math.sqrt(self.args.hidden_dim)
            embedded_context, _ = self.text_embedder(embedded_context)
            encoder_outputs = self.text_encoder(embedded_context.permute(
                1, 0, 2), src_mask).permute(1, 0, 2)  # [B, T, D]
            encoder_outputs[:, 1] = encoder_outputs[:, 1] + image_features
            encoder_outputs = torch.cat(
                (encoded_objects, encoder_outputs), dim=1)
            decoder_outputs = self.decode_rnn(
                encoder_outputs, encoder_outputs[:, 1], target
            )

        if self.args.variant == "transformer-oca-lt":
            encoded_objects, _ = self.encode_object_features(
                rcnn_features, rcnn_locations)
            src_mask = self.text_encoder.generate_square_subsequent_mask(
                answers.shape[-1]).to(self.args.device)
            embedded_context = self.embedding(
                answers) * math.sqrt(self.args.hidden_dim)
            embedded_context, _ = self.text_embedder(embedded_context)
            encoder_outputs = self.text_encoder(embedded_context.permute(
                1, 0, 2), src_mask).permute(1, 0, 2)  # [B, T, D]
            encoder_outputs[:, 1] = encoder_outputs[:, 1] + image_features
            encoder_outputs = torch.cat(
                (encoded_objects, encoder_outputs), dim=1)
            image_mask = torch.zeros(
                encoded_objects.shape[0], 1, 36, device=self.args.device).bool()
            src_mask = torch.cat(
                (image_mask, generate_pad_mask(answers)), dim=-1)
            decoder_outputs, z_logit = self.decoder_T(
                encoder_outputs, target, image_features, encoder_outputs[:, 1], src_mask
            )

        if self.args.variant == "lstm-latentNorm-oc":
            category_features = self.category_embedding(category)
            cat_features = torch.cat((category_features, image_features), -1)
            cat_img = self.category_image_proj(cat_features)
            encoded_objects, _ = self.encode_object_features(
                rcnn_features, rcnn_locations)
            z, kld_loss = self.latent_layer(encoded_objects)
            decoder_outputs = self.decode_rnn(
                z, cat_img, target)  # [B, T, D]

        if self.args.variant == "lstm-latent-oca-lt":
            category_features = self.category_embedding(category)
            cat_features = torch.cat((category_features, image_features), -1)
            cat_img = self.category_image_proj(cat_features)
            encoded_objects, _ = self.encode_object_features(
                rcnn_features, rcnn_locations)
            src_mask = self.text_encoder.generate_square_subsequent_mask(
                answers.shape[-1]).to(self.args.device)
            embedded_context = self.embedding(
                answers) * math.sqrt(self.args.hidden_dim)
            embedded_context, _ = self.text_embedder(embedded_context)
            encoder_outputs = self.text_encoder(embedded_context.permute(
                1, 0, 2), src_mask).permute(1, 0, 2)  # [B, T, D]
            encoder_outputs[:, 0] = encoder_outputs[:, 0] + cat_img
            z, kld_loss = self.latent_layer(
                cat_img, encoder_outputs[:, 0])
            decoder_outputs = self.decode_rnn(
                encoded_objects, z, target)  # [B, T, D]

        if self.args.variant == "transformer-latentNorm-oc":
            category_features = self.category_embedding(category)
            cat_features = torch.cat((category_features, image_features), -1)
            cat_img = self.category_image_proj(cat_features)
            encoded_objects, image_pad_mask = self.encode_object_features(
                rcnn_features, rcnn_locations)
            if self.latent_transformer:
                z, kld_loss = self.latent_layer(encoded_objects)
                decoder_outputs, z_logit = self.decoder_T(
                    z, target, cat_img, cat_img, image_pad_mask
                )
            else:
                z, kld_loss = torch.tensor(0), torch.tensor(0)
                decoder_outputs, z_logit = self.decoder_T(
                    encoded_objects, target, cat_img, cat_img, image_pad_mask
                )

        if self.args.variant == "transformer-latent-oca-lt":
            category_features = self.category_embedding(category)
            cat_features = torch.cat((category_features, image_features), -1)
            cat_img = self.category_image_proj(cat_features)
            encoded_objects, image_pad_mask = self.encode_object_features(
                rcnn_features, rcnn_locations)
            src_mask = self.text_encoder.generate_square_subsequent_mask(
                answers.shape[-1]).to(self.args.device)
            embedded_context = self.embedding(
                answers) * math.sqrt(self.args.hidden_dim)
            embedded_context, _ = self.text_embedder(embedded_context)
            encoder_outputs = self.text_encoder(embedded_context.permute(
                1, 0, 2), src_mask).permute(1, 0, 2)  # [B, T, D]
            encoder_outputs[:, 0] = encoder_outputs[:, 0] + image_features
            z, kld_loss = self.latent_layer(
                cat_img, encoder_outputs[:, 0])
            encoder_outputs = torch.cat(
                (encoded_objects, encoder_outputs), dim=1)
            src_mask = torch.cat(
                (image_pad_mask, generate_pad_mask(answers)), dim=-1)
            if self.latent_transformer:
                decoder_outputs, z_logit = self.decoder_T(
                    encoder_outputs, target, cat_img, z, src_mask
                )
            else:
                decoder_outputs, z_logit = self.decoder_T(
                    encoder_outputs, target, cat_img, 0, src_mask
                )

                # z-path. transformer_posteriors is a tuple: (mean_posterior, logvar_posterior)
                # encoder_outputs, transformer_kld_loss, z, transformer_posteriors, src_mask = self.answer_encoder(
                #     answers, response, image_features, rcnn_features, rcnn_locations)
                # if self.latent_transformer:
                #     z = self.latent_projection(z)
                # else:
                #     z = encoder_outputs[:, 0]
                # decoder_outputs, z_logit = self.decoder(
                # encoded_objects, target, image_features, image_features, image_pad_mask.unsqueeze(1))

                # decoder_outputs = self.decode_rnn(
                #     encoder_outputs, z, target)  # [B, T, D]
                # output, z_logit = self.decoder(
                #     encoder_outputs, target, image_features, z, src_mask)

                # if self.latent_transformer:  # experiement without requiring the latent mode enabled?
                #     reconstructed_image_features = self.image_reconstructor(
                #         encoder_outputs[:, 0] + z)
                # else:
                #     reconstructed_image_features = self.image_reconstructor(
                #         encoder_outputs[:, 0])

        reconstructed_image_features = None

        # return output, z_logit, transformer_kld_loss, (image_features, reconstructed_image_features)
        return decoder_outputs, None, kld_loss, (image_features, reconstructed_image_features)

    def decode_greedy(self, images, answers, target, rcnn_features, rcnn_locations, max_decode_length=50):
        image_features = self.encoder_cnn(images)

        if self.args.variant == "lstm-baseline":
            decoder_outputs = self.decode_rnn(
                image_features.unsqueeze(1), image_features, target, 0)

        if self.args.variant == "transformer-baseline":
            decoder_input_mask = torch.zeros(
                images.shape[0], 1, 1, device=self.args.device).bool()
            encoder_outputs = image_features.unsqueeze(1)
            z = image_features

        if self.args.variant == "lstm-o":
            encoded_objects, _ = self.encode_object_features(
                rcnn_features, rcnn_locations)
            decoder_outputs = self.decode_rnn(
                encoded_objects, image_features, target, 0)  # [B, T, D]

        if self.args.variant == "transformer-o":
            encoded_objects, image_pad_mask = self.encode_object_features(
                rcnn_features, rcnn_locations)
            encoder_outputs, decoder_input_mask = encoded_objects, image_pad_mask
            z = image_features

        if self.args.variant == "lstm-c":
            category_features = self.category_embedding(answers)
            cat_features = torch.cat((category_features, image_features), -1)
            cat_img = self.category_image_proj(cat_features)
            decoder_outputs = self.decode_rnn(
                cat_img.unsqueeze(1), cat_img, target, 0)  # [B, T, D]

        if self.args.variant == "transformer-c":
            src_mask = self.text_encoder.generate_square_subsequent_mask(
                answers.shape[-1]).to(self.args.device)
            embedded_context = self.embedding(
                answers) * math.sqrt(self.args.hidden_dim)
            encoder_outputs = self.text_encoder(embedded_context.permute(
                1, 0, 2), src_mask).permute(1, 0, 2)  # [B, T, D]
            decoder_input_mask = generate_pad_mask(answers)
            z = encoder_outputs[:, 1]

        if self.args.variant == "lstm-oc":
            category_features = self.category_embedding(answers)
            cat_features = torch.cat((category_features, image_features), -1)
            cat_img = self.category_image_proj(cat_features)
            encoded_objects, _ = self.encode_object_features(
                rcnn_features, rcnn_locations)
            decoder_outputs = self.decode_rnn(
                encoded_objects, cat_img, target, 0)  # [B, T, D]

        if self.args.variant == "transformer-oc":
            category_features = self.category_embedding(answers)
            cat_features = torch.cat((category_features, image_features), -1)
            cat_img = self.category_image_proj(cat_features)
            encoded_objects, image_pad_mask = self.encode_object_features(
                rcnn_features, rcnn_locations)
            decoder_input_mask, encoder_outputs, image_features, z = image_pad_mask, encoded_objects, cat_img, cat_img

        if self.args.variant == "lstm-ca":
            src_mask = self.text_encoder.generate_square_subsequent_mask(
                answers.shape[-1]).to(self.args.device)
            embedded_context = self.embedding(
                answers) * math.sqrt(self.args.hidden_dim)
            encoder_outputs = self.text_encoder(embedded_context.permute(
                1, 0, 2), src_mask).permute(1, 0, 2)  # [B, T, D]
            encoder_outputs[:, 1] = encoder_outputs[:, 1] + image_features
            decoder_outputs = self.decode_rnn(
                encoder_outputs, encoder_outputs[:, 1], target, 0
            )

        if self.args.variant == "transformer-ca":
            src_mask = self.text_encoder.generate_square_subsequent_mask(
                answers.shape[-1]).to(self.args.device)
            embedded_context = self.embedding(
                answers) * math.sqrt(self.args.hidden_dim)
            encoder_outputs = self.text_encoder(embedded_context.permute(
                1, 0, 2), src_mask).permute(1, 0, 2)  # [B, T, D]
            decoder_input_mask = generate_pad_mask(answers)
            z = encoder_outputs[:, 1]

        if self.args.variant == "lstm-oca":
            encoded_objects, _ = self.encode_object_features(
                rcnn_features, rcnn_locations)
            src_mask = self.text_encoder.generate_square_subsequent_mask(
                answers.shape[-1]).to(self.args.device)
            embedded_context = self.embedding(
                answers) * math.sqrt(self.args.hidden_dim)
            encoder_outputs = self.text_encoder(embedded_context.permute(
                1, 0, 2), src_mask).permute(1, 0, 2)  # [B, T, D]
            encoder_outputs[:, 1] = encoder_outputs[:, 1] + image_features
            encoder_outputs = torch.cat(
                (encoded_objects, encoder_outputs), dim=1)
            decoder_outputs = self.decode_rnn(
                encoder_outputs, encoder_outputs[:, 1], target, 0
            )

        if self.args.variant == "transformer-oca":
            encoded_objects, _ = self.encode_object_features(
                rcnn_features, rcnn_locations)
            src_mask = self.text_encoder.generate_square_subsequent_mask(
                answers.shape[-1]).to(self.args.device)
            embedded_context = self.embedding(
                answers) * math.sqrt(self.args.hidden_dim)
            encoder_outputs = self.text_encoder(embedded_context.permute(
                1, 0, 2), src_mask).permute(1, 0, 2)  # [B, T, D]
            encoder_outputs[:, 1] = encoder_outputs[:, 1] + image_features
            encoder_outputs = torch.cat(
                (encoded_objects, encoder_outputs), dim=1)
            image_mask = torch.zeros(
                encoded_objects.shape[0], 1, 36, device=self.args.device).bool()
            decoder_input_mask = torch.cat(
                (image_mask, generate_pad_mask(answers)), dim=-1)
            z = encoder_outputs[:, 1]

        if self.args.variant == "lstm-oca-lt":
            encoded_objects, _ = self.encode_object_features(
                rcnn_features, rcnn_locations)
            src_mask = self.text_encoder.generate_square_subsequent_mask(
                answers.shape[-1]).to(self.args.device)
            embedded_context = self.embedding(
                answers) * math.sqrt(self.args.hidden_dim)
            embedded_context, _ = self.text_embedder(embedded_context)
            encoder_outputs = self.text_encoder(embedded_context.permute(
                1, 0, 2), src_mask).permute(1, 0, 2)  # [B, T, D]
            encoder_outputs[:, 1] = encoder_outputs[:, 1] + image_features
            encoder_outputs = torch.cat(
                (encoded_objects, encoder_outputs), dim=1)
            decoder_outputs = self.decode_rnn(
                encoder_outputs, encoder_outputs[:, 1], target
            )

        if self.args.variant == "transformer-oca-lt":
            encoded_objects, _ = self.encode_object_features(
                rcnn_features, rcnn_locations)
            src_mask = self.text_encoder.generate_square_subsequent_mask(
                answers.shape[-1]).to(self.args.device)
            embedded_context = self.embedding(
                answers) * math.sqrt(self.args.hidden_dim)
            embedded_context, _ = self.text_embedder(embedded_context)
            encoder_outputs = self.text_encoder(embedded_context.permute(
                1, 0, 2), src_mask).permute(1, 0, 2)  # [B, T, D]
            encoder_outputs[:, 1] = encoder_outputs[:, 1] + image_features
            encoder_outputs = torch.cat(
                (encoded_objects, encoder_outputs), dim=1)
            image_mask = torch.zeros(
                encoded_objects.shape[0], 1, 36, device=self.args.device).bool()
            decoder_input_mask = torch.cat(
                (image_mask, generate_pad_mask(answers)), dim=-1)
            z = encoder_outputs[:, 1]

        if self.args.variant == "lstm-latentNorm-oc":
            category_features = self.category_embedding(answers)
            cat_features = torch.cat((category_features, image_features), -1)
            cat_img = self.category_image_proj(cat_features)
            encoded_objects, _ = self.encode_object_features(
                rcnn_features, rcnn_locations)
            z, kld_loss = self.latent_layer(encoded_objects)
            decoder_outputs = self.decode_rnn(
                z, cat_img, target)  # [B, T, D]

        if self.args.variant == "lstm-latent-oca-lt":
            category_features = self.category_embedding(answers)
            cat_features = torch.cat((category_features, image_features), -1)
            cat_img = self.category_image_proj(cat_features)
            encoded_objects, _ = self.encode_object_features(
                rcnn_features, rcnn_locations)
            z, kld_loss = self.latent_layer(
                cat_img, None)
            decoder_outputs = self.decode_rnn(
                encoded_objects, z, target)  # [B, T, D]

        if self.args.variant == "transformer-latentNorm-oc":
            category_features = self.category_embedding(answers)
            cat_features = torch.cat((category_features, image_features), -1)
            cat_img = self.category_image_proj(cat_features)
            encoded_objects, image_pad_mask = self.encode_object_features(
                rcnn_features, rcnn_locations)
            if self.latent_transformer:
                z, kld_loss = self.latent_layer(encoded_objects)
                decoder_input_mask, encoder_outputs, image_features, z = image_pad_mask, z, cat_img, cat_img
            else:
                decoder_input_mask, encoder_outputs, image_features, z = image_pad_mask, encoded_objects, cat_img, cat_img

        if self.args.variant == "transformer-latent-oca-lt":
            category_features = self.category_embedding(answers)
            cat_features = torch.cat((category_features, image_features), -1)
            cat_img = self.category_image_proj(cat_features)
            encoded_objects, image_pad_mask = self.encode_object_features(
                rcnn_features, rcnn_locations)
            if self.latent_transformer:
                z, kld_loss = self.latent_layer(cat_img, None)
                decoder_input_mask, encoder_outputs, image_features, z = image_pad_mask, encoded_objects, cat_img, z
            else:
                decoder_input_mask, encoder_outputs, image_features, z = image_pad_mask, encoded_objects, cat_img, 0

            # encoded_objects, image_pad_mask = self.image_encoder(
            #     rcnn_features, rcnn_locations)  # encoded_objects [T, B, D]
            # encoded_objects = encoded_objects.permute(1, 0, 2)  # [B, T, D]
            # category_features = self.category_embedding(answers)
            # # encoder_outputs, _, z, _, src_mask = self.answer_encoder(
            # #     answers, None, image_features, rcnn_features, rcnn_locations)
            # cat_features = torch.cat((category_features, image_features), -1)
            # cat_img = self.category_image_proj(cat_features)

            # src_mask = generate_pad_mask(answers)

            # if self.latent_transformer:
            #     z = self.latent_projection(z)
            # else:
            #     z = encoder_outputs[:, 0]

            # decoder_outputs = self.decode_rnn(encoder_outputs, z, target, 0)

        if self.args.variant.startswith("lstm"):
            decoder_tokens: Tensor = torch.argmax(decoder_outputs, dim=-1)
            decoded = []
            for batch in decoder_tokens:
                decoded_words = []
                for word in batch:
                    if word.item() == self.vocab.word2idx[self.vocab.SYM_EOS]:
                        decoded_words.append("<end>")
                    else:
                        decoded_words.append(self.vocab.idx2word[word.item()])
                decoded.append(decoded_words)

            decoded = [" ".join(sentence) for sentence in decoded]
            return decoded

        # encoder_outputs = image_features.unsqueeze(1)
        # src_mask = torch.zeros(
        #     encoder_outputs.shape[0], 1, 1, device=self.args.device).bool()

        if self.args.variant.startswith("transformer"):
            ys = torch.ones(answers.shape[0], 1).fill_(
                self.vocab.word2idx[self.vocab.SYM_PAD]).long().to(self.args.device)
            top_args = torch.zeros(
                answers.shape[0], max_decode_length+1, 6).to(self.args.device)
            top_args_vals = torch.zeros(
                answers.shape[0], max_decode_length+1, 6).to(self.args.device)

            decoded_words = []
            for i in range(max_decode_length + 1):
                pred_targets_logit = self.decoder_T.inference_forward(
                    encoder_outputs, ys, image_features, z, decoder_input_mask)
                _, pred_next_word = torch.max(pred_targets_logit[:, -1], dim=1)
                top_6_vals, top_6_indicies = torch.topk(
                    torch.nn.functional.softmax(pred_targets_logit[:, -1], -1), 6, dim=1)

                decoded_words.append(['<end>' if token.item() == self.vocab.word2idx[self.vocab.SYM_EOS]
                                      else self.vocab.idx2word[token.item()] for token in pred_next_word.view(-1)])

                ys = torch.cat([ys, pred_next_word.unsqueeze(1)], dim=1)
                top_args[:, i] = top_6_indicies
                top_args_vals[:, i] = top_6_vals

            sentence = []
            for _, row in enumerate(np.transpose(decoded_words)):
                st = ''
                for e in row:
                    if e == '<end>':
                        break
                    else:
                        st += e + ' '
                sentence.append(st)
            return sentence
