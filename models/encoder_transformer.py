
import torch
from models.transformer_layers import Encoder, generate_pad_mask, ImageTransformerEncoder
from torch import nn


class GVTransformerEncoder(nn.Module):
    def __init__(self, embedding, latent_layer, latent_transformer, args):
        super().__init__()

        self.embedding = embedding
        self.latent_transformer = latent_transformer
        self.latent_layer = latent_layer

        self.encoder = Encoder(args.emb_dim, args.hidden_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                               total_key_depth=args.hidden_dim, total_value_depth=args.hidden_dim,
                               filter_size=args.pwffn_dim, input_dropout=args.dropout, layer_dropout=args.dropout,
                               attention_dropout=args.dropout, relu_dropout=args.dropout)

        self.r_encoder = Encoder(args.emb_dim, args.hidden_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                                 total_key_depth=args.hidden_dim, total_value_depth=args.hidden_dim,
                                 filter_size=args.pwffn_dim, input_dropout=args.dropout, layer_dropout=args.dropout,
                                 attention_dropout=args.dropout, relu_dropout=args.dropout)

        self.image_encoder = ImageTransformerEncoder(args)
        self.text_image_alignment = Encoder(args.emb_dim, args.hidden_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                                            total_key_depth=args.hidden_dim, total_value_depth=args.hidden_dim,
                                            filter_size=args.pwffn_dim, input_dropout=args.dropout, layer_dropout=args.dropout,
                                            attention_dropout=args.dropout, relu_dropout=args.dropout)

    def forward(self, context, response, image_features, rcnn_features, rcnn_locations):
        if response is not None:
            res_mask = generate_pad_mask(response)
            embedded_response = self.embedding(response)
            response_encoder_outputs = self.r_encoder(
                embedded_response, res_mask)
        else:
            response_encoder_outputs = None

        src_mask = generate_pad_mask(context)
        embedded_context = self.embedding(context)
        encoder_outputs = self.encoder(embedded_context, src_mask)

        encoded_images, image_pad_mask = self.image_encoder(
            rcnn_features, rcnn_locations)  # encoded_images [T, B, D]
        encoded_images = encoded_images.permute(1, 0, 2)
        image_pad_mask = image_pad_mask.unsqueeze(1)
        encoder_outputs[:, 0] = encoder_outputs[:, 0] + image_features
        encoder_outputs = torch.cat((encoder_outputs, encoded_images), dim=1)

        return_mask = torch.cat((src_mask, image_pad_mask), dim=-1)

        # encoder_outputs = self.text_image_alignment(
        #     encoder_outputs, return_mask)

        # TEST WHETHER THIS IS BENEFICIAL/DETRIMENTAL
        kld_loss, z, posteriors = None, None, None
        if self.latent_transformer:
            if response is None:
                kld_loss, z, posteriors = self.latent_layer(
                    encoder_outputs[:, 0], None)
            else:
                kld_loss, z, posteriors = self.latent_layer(
                    encoder_outputs[:, 0], response_encoder_outputs[:, 0])

        return encoder_outputs, kld_loss, z, posteriors, return_mask

# from models.transformer_layers import Encoder, generate_pad_mask
# from torch import nn

# class GVTransformerEncoder(nn.Module):
#     def __init__(self, embedding, latent_layer, latent_transformer, args):
#         super().__init__()

#         self.embedding = embedding
#         self.latent_transformer = latent_transformer
#         self.latent_layer = latent_layer

#         self.encoder = Encoder(args.emb_dim, args.hidden_dim, num_layers=args.num_layers, num_heads=args.num_heads,
#                                 total_key_depth=args.hidden_dim, total_value_depth=args.hidden_dim,
#                                 filter_size=args.pwffn_dim)

#         self.r_encoder = Encoder(args.emb_dim, args.hidden_dim, num_layers=args.num_layers, num_heads=args.num_heads,
#                                 total_key_depth=args.hidden_dim, total_value_depth=args.hidden_dim,
#                                 filter_size=args.pwffn_dim)


#     def forward(self, context, response, image_features):
#         res_mask = generate_pad_mask(response)
#         embedded_response = self.embedding(response)
#         response_encoder_outputs = self.r_encoder(embedded_response, res_mask)

#         src_mask = generate_pad_mask(context)
#         embedded_context = self.embedding(context)
#         encoder_outputs = self.encoder(embedded_context, src_mask)

#         encoder_outputs[:, 0] = encoder_outputs[:, 0] + image_features

#         return encoder_outputs, response_encoder_outputs, src_mask
