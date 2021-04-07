"""Contains code for the IQ model.
"""

from math import log
import os

from transformers import BertGenerationDecoder, BertGenerationEncoder, BertGenerationConfig
from transformers.utils.dummy_tokenizers_objects import BertTokenizerFast
# from utils import vocab
from models import transformer_layers
from models.decoder_transformer import GVTransformerDecoder
from models.transformer_layers import Latent, generate_pad_mask
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
    def __init__(self, latent_transformer, tokenizer: BertTokenizerFast, args, num_att_layers=2):
        super(IQ, self).__init__()
        self.tokenizer = tokenizer
        self.latent_transformer = latent_transformer
        self.args = args

        # Setup image encoder.
        self.encoder_cnn = EncoderCNN(args)

        self.latent_layer = Latent(args)
        self.latent_projection = nn.Linear(args.latent_dim, args.hidden_dim)

        self.segment_embedding = nn.Embedding(2, args.hidden_dim)

        config = BertGenerationConfig.from_pretrained("bert-large-uncased", hidden_size=args.hidden_dim, num_hidden_layers=args.num_layers, num_attention_heads=args.num_heads, intermediate_size=args.pwffn_dim, hidden_dropout_prob=args.dropout,  bos_token_id=101, eos_token_id=102)

        self.answer_encoder = BertGenerationEncoder.from_pretrained("bert-large-uncased", config=config)
        self.response_encoder = BertGenerationEncoder.from_pretrained("bert-large-uncased", config=config)
        # config.is_decoder = True
        # config.add_cross_attention = True
        # self.decoder = BertGenerationDecoder.from_pretrained("bert-large-uncased", config=config)

        self.answer_encoder.resize_token_embeddings(len(self.tokenizer))
        self.response_encoder.resize_token_embeddings(len(self.tokenizer))
        # self.decoder.resize_token_embeddings(len(self.tokenizer))
        # self.z_classifier = nn.Linear(args.hidden_dim, len(tokenizer))

        self.decoder = GVTransformerDecoder(self.answer_encoder.embeddings, self.latent_transformer, self.tokenizer, args)




        # Setup image reconstruction.
        self.image_reconstructor = MLP(
                args.hidden_dim, args.pwffn_dim, args.hidden_dim,
                num_layers=num_att_layers)


    def switch_GVT_train_mode(self, new_mode):
        self.latent_transformer = new_mode
        self.answer_encoder.latent_transformer = new_mode
        self.decoder.latent_transformer = new_mode


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
        image_features = self.encoder_cnn(images)

        answer_word_embeddings = self.answer_encoder.embeddings(input_ids=answers["input_ids"], position_ids=None, inputs_embeds=None, past_key_values_length=0)
        # answer_segment_embeddings = self.segment_embedding(answers["token_type_ids"])
        # answer_embeddings = answer_word_embeddings + answer_segment_embeddings
        answer_encoding = self.answer_encoder(inputs_embeds=answer_word_embeddings)
        answer_encoding = answer_encoding.last_hidden_state

        response_word_embeddings = self.answer_encoder.embeddings(input_ids=response["input_ids"], position_ids=None, inputs_embeds=None, past_key_values_length=0)
        # response_segment_embeddings = self.segment_embedding(response["token_type_ids"])
        # response_embeddings = response_word_embeddings + response_segment_embeddings
        response_encoding = self.response_encoder(inputs_embeds=response_word_embeddings)
        response_encoding = response_encoding.last_hidden_state

        answer_encoding[:, 0] = answer_encoding[:, 0] + image_features
        kld, z = torch.tensor([0]).to(self.args.device), torch.tensor([0]).to(self.args.device)
        if self.latent_transformer:
            kld, z, posteriors = self.latent_layer(answer_encoding[:, 0], response_encoding[:, 0])
            z = self.latent_projection(z)
        output, z_logit = self.decoder(answer_encoding, target, image_features, z, None)
        

        # target_word_embeddings = self.decoder.bert.embeddings(input_ids=target["input_ids"], position_ids=None, inputs_embeds=None, past_key_values_length=0)
        # target_word_embeddings[:, 0] = target_word_embeddings[:, 0] + image_features + z # z = 0 when NOT in latent_transformer mode.
        # output = self.decoder(inputs_embeds=target_word_embeddings, encoder_hidden_states=answer_encoding, labels=target["input_ids"])

        # decoder_loss, decoder_logits = output.loss, output.logits

        z_logit = None
        if self.latent_transformer:
            z_logit = self.z_classifier(z + image_features)
            reconstructed_image_features = self.image_reconstructor(answer_encoding[:, 0] + z)
        else:
            reconstructed_image_features = self.image_reconstructor(answer_encoding[:, 0])


        return output, z_logit, kld, (image_features, reconstructed_image_features)


    def decode_greedy(self, images, categorys, max_decode_length = 50):
        image_features = self.encoder_cnn(images)
        category_word_embeddings = self.answer_encoder.embeddings(input_ids=categorys["input_ids"], position_ids=None, inputs_embeds=None, past_key_values_length=0)
        segment_embeddings = self.segment_embedding(categorys["token_type_ids"])
        category_embeddings = category_word_embeddings + segment_embeddings
        category_encoding = self.answer_encoder(inputs_embeds=category_embeddings)
        category_encoding = category_encoding.last_hidden_state

        category_encoding[:, 0] = category_encoding[:, 0] + image_features # TEST THISSS

        z = 0
        if self.latent_transformer:
            _, z, _ = self.latent_layer(category_encoding[:, 0], None)
            z = self.latent_projection(z)

        ys = torch.ones(self.args.batch_size, 1).fill_(101).long().to(self.args.device)
        # top_args = torch.zeros(answers.shape[0], max_decode_length+1, 6).to(self.args.device)
        # top_args_vals = torch.zeros(answers.shape[0], max_decode_length+1, 6).to(self.args.device)

        decoded_words = []
        for i in range(max_decode_length + 1):
            pred_targets_logit = self.decoder.inference_forward(category_encoding, ys, image_features, z, None)
            _, pred_next_word = torch.max(pred_targets_logit[:, -1], dim=1)
            # top_6_vals, top_6_indicies = torch.topk(torch.nn.functional.softmax(pred_targets_logit[:, -1], -1), 6, dim=1)

            decoded_words.append(['<end>' if token.item() == 102 else self.tokenizer.convert_ids_to_tokens(token.item()) for token in pred_next_word.view(-1)])

            ys = torch.cat([ys, pred_next_word.unsqueeze(1)], dim=1)
            # top_args[:, i] = top_6_indicies
            # top_args_vals[:, i] = top_6_vals

        sentence = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ''
            for e in row:
                if e == '<end>': break
                else: st+= e + ' '
            sentence.append(st)
        return sentence
