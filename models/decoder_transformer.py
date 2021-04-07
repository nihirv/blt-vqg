from models.transformer_layers import Decoder, generate_pad_mask
from torch import nn
import torch

class GVTransformerDecoder(nn.Module):
    def __init__(self, embedding, latent_transformer, vocab_size, vocab, args):
        super().__init__()

        self.embedding = embedding
        self.latent_transformer = latent_transformer
        self.vocab = vocab
        self.args = args

        self.decoder = Decoder(args.emb_dim, hidden_size = args.hidden_dim, num_layers=args.num_layers, num_heads=args.num_heads, 
                            total_key_depth=args.hidden_dim, total_value_depth=args.hidden_dim,
                            filter_size=args.pwffn_dim, device=args.device, input_dropout=args.dropout, layer_dropout=args.dropout, 
                                attention_dropout=args.dropout, relu_dropout=args.dropout)


        self.output = nn.Linear(args.hidden_dim, vocab_size)
        self.z_classifier = nn.Linear(args.hidden_dim, vocab_size)

    def forward(self, encoder_outputs, target, image_features, z, src_mask):

        sos_token = torch.LongTensor([self.vocab.word2idx[self.vocab.SYM_SOQ]] * encoder_outputs.size(0)).unsqueeze(1)
        sos_token = sos_token.to(self.args.device)

        target_shifted = torch.cat((sos_token, target[:, :-1]), 1)
        trg_key_padding_mask = generate_pad_mask(target_shifted)
        target_embedding = self.embedding(target_shifted)

        target_embedding[:, 0] = target_embedding[:, 0] + image_features # z = 0 if we're pretraining
        z_logit = None
        if self.latent_transformer:
            target_embedding[:, 0] = target_embedding[:, 0] + z
            z_logit = self.z_classifier(z + image_features)

        # decoder_outputs = self.transformer_decoder(target_embedding, encoder_outputs, trg_mask, src_mask.unsqueeze(1))
        decoder_outputs, _ = self.decoder(target_embedding, encoder_outputs, (src_mask, trg_key_padding_mask))

        output = self.output(decoder_outputs)
        return output, z_logit

    def inference_forward(self, encoder_outputs, inference_input, image_features, z, src_mask):
        trg_key_padding_mask = generate_pad_mask(inference_input)
        pred_targets_embedding = self.embedding(inference_input)
        pred_targets_embedding[:, 0] = pred_targets_embedding[:, 0] + z + image_features
        decoder_outputs, _ = self.decoder(pred_targets_embedding, encoder_outputs, (src_mask, trg_key_padding_mask))
        return self.output(decoder_outputs)

# from models.transformer_layers import Decoder, generate_pad_mask
# from torch import nn
# import torch
# import torch.nn.functional as F

# class GVTransformerDecoder(nn.Module):
#     def __init__(self, embedding, latent_transformer, vocab_size, vocab, args):
#         super().__init__()

#         self.embedding = embedding
#         self.latent_transformer = latent_transformer
#         self.vocab = vocab
#         self.args = args

#         self.decoder = Decoder(args.emb_dim, hidden_size = args.hidden_dim, num_layers=args.num_layers, num_heads=args.num_heads, 
#                             total_key_depth=args.hidden_dim, total_value_depth=args.hidden_dim,
#                             filter_size=args.pwffn_dim, device=args.device, input_dropout=args.dropout, layer_dropout=args.dropout, 
#                                 attention_dropout=args.dropout, relu_dropout=args.dropout)


#         self.deep_question_alignment = nn.Sequential(
#             nn.Linear(args.hidden_dim, args.hidden_dim),
#             nn.Sigmoid()
#         )

#         self.output = nn.Linear(args.hidden_dim, vocab_size)
#         self.z_classifier = nn.Linear(args.hidden_dim, vocab_size)

#     # def deep_questioner(self, encoder_outputs, feature_maps, mask):
#     #     deep_q_image = self.deep_question_alignment(feature_maps) # [B, T_i, d], [B, 49, 1024]
#     #     deep_q_text = self.deep_question_alignment(encoder_outputs) # [B, T_t, d]
#     #     affinity_matrix_1 = torch.bmm(deep_q_image, deep_q_text.permute(0, 2, 1)) # [B, T_i, T_t]
#     #     max_col_affinity_matrix_1, _ = affinity_matrix_1.max(-1)
#     #     max_col_affinity_matrix_1 = F.softmax(max_col_affinity_matrix_1, dim=-1) # [B, T_i]
#     #     contextual_representation_1 = feature_maps * max_col_affinity_matrix_1.unsqueeze(-1) # [B, T_i, d]

#     #     deep_q_cr1 = self.deep_question_alignment(contextual_representation_1) # [B, T_i, d]
#     #     affinity_matrix_2 = torch.bmm(deep_q_image, deep_q_cr1.permute(0, 2, 1))
#     #     max_col_affinity_matrix_2, _ = affinity_matrix_2.max(-1)
#     #     max_col_affinity_matrix_2 = F.softmax(max_col_affinity_matrix_2, dim=-1)
#     #     contextual_representation_2 = feature_maps * max_col_affinity_matrix_2.unsqueeze(-1)

#     #     ### Mask(t)ing
#     #     len_hw = feature_maps.shape[1]
#     #     no_pad_for_images_mask = torch.zeros(feature_maps.shape[0], 1, len_hw).to(self.args.device)
#     #     new_mask = torch.cat((no_pad_for_images_mask.bool(), mask), -1)
#     #     ###
#     #     return contextual_representation_2, new_mask

#     def forward(self, encoder_outputs, target, image_features, feature_maps, z, src_mask):

#         # conextual_representation, src_mask = self.deep_questioner(encoder_outputs, feature_maps, src_mask)

#         sos_token = torch.LongTensor([self.vocab.word2idx[self.vocab.SYM_SOQ]] * encoder_outputs.size(0)).unsqueeze(1)
#         sos_token = sos_token.to(self.args.device)

#         target_shifted = torch.cat((sos_token, target[:, :-1]), 1)
#         trg_key_padding_mask = generate_pad_mask(target_shifted)
#         target_embedding = self.embedding(target_shifted)

#         target_embedding[:, 0] = target_embedding[:, 0] + image_features # z = 0 if we're pretraining
#         z_logit = None
#         if self.latent_transformer:
#             target_embedding[:, 0] = target_embedding[:, 0] + z
#             z_logit = self.z_classifier(z + image_features)

#         # image_cat_encoder_outputs = torch.cat((conextual_representation, encoder_outputs), 1)

#         decoder_outputs, _ = self.decoder(target_embedding, encoder_outputs, (None, trg_key_padding_mask))
#         # decoder_outputs, _ = self.decoder(target_embedding, image_cat_encoder_outputs, (src_mask, trg_key_padding_mask))

#         output = self.output(decoder_outputs)
#         return output, z_logit

#     def inference_forward(self, encoder_outputs, inference_input, image_features, feature_maps, z, src_mask):
#         # conextual_representation, src_mask = self.deep_questioner(encoder_outputs, feature_maps, src_mask)
#         trg_key_padding_mask = generate_pad_mask(inference_input)
#         pred_targets_embedding = self.embedding(inference_input)
#         pred_targets_embedding[:, 0] = pred_targets_embedding[:, 0] + z + image_features
#         # pred_image_cat_encoder_outputs = torch.cat((conextual_representation, encoder_outputs), 1)
#         # decoder_outputs, _ = self.decoder(pred_targets_embedding, pred_image_cat_encoder_outputs, (src_mask, trg_key_padding_mask))
#         decoder_outputs, _ = self.decoder(pred_targets_embedding, encoder_outputs, (None, trg_key_padding_mask))
#         return self.output(decoder_outputs)