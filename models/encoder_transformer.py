import torch
from models.transformer_layers import Encoder, generate_pad_mask
from torch import nn

class GVTransformerEncoder(nn.Module):
    def __init__(self, embedding, latent_layer, latent_transformer, args):
        super().__init__()

        self.embedding = embedding
        self.latent_transformer = latent_transformer
        self.latent_layer = latent_layer
        self.args = args
        
        self.encoder = Encoder(args.emb_dim, args.hidden_dim, num_layers=args.enc_num_layers, num_heads=args.num_heads, 
                                total_key_depth=args.hidden_dim, total_value_depth=args.hidden_dim,
                                filter_size=args.pwffn_dim)
        
        self.r_encoder = Encoder(args.emb_dim, args.hidden_dim, num_layers=args.enc_num_layers, num_heads=args.num_heads, 
                                total_key_depth=args.hidden_dim, total_value_depth=args.hidden_dim,
                                filter_size=args.pwffn_dim)
        
        # self.category_segment_encoding = nn.Embedding(1, args.hidden_dim)
        # self.answer_segment_encoding = nn.Embedding(1, args.hidden_dim)
        # self.question_segment_encoding = nn.Embedding(1, args.hidden_dim)

    def forward(self, context, response, image_features):
        # one_tensor = torch.LongTensor([0]).to(self.args.device).detach()
        # category_encoding_vector = self.category_segment_encoding(one_tensor)
        # answer_encoding_vector = self.answer_segment_encoding(one_tensor)
        # question_encoding_vector = self.question_segment_encoding(one_tensor)

        res_mask = generate_pad_mask(response)
        embedded_response = self.embedding(response)
        # embedded_response[:, :2] = embedded_response[:, :2] + category_encoding_vector
        # embedded_response[:, 2:] = embedded_response[:, 2:] + question_encoding_vector
        response_encoder_outputs = self.r_encoder(embedded_response, res_mask)

        src_mask = generate_pad_mask(context)
        embedded_context = self.embedding(context)
        # embedded_context[:, :2] = embedded_context[:, :2] + category_encoding_vector
        # embedded_context[:, 2:] = embedded_context[:, 2:] + answer_encoding_vector
        encoder_outputs = self.encoder(embedded_context, src_mask)


        # TEST WHETHER THIS IS BENEFICIAL/DETRIMENTAL
        encoder_outputs[:, 0] = encoder_outputs[:, 0] + image_features
        z_kld_loss, z, priors, posteriors = None, None, None, None
        if self.latent_transformer:
            z_kld_loss, z, priors, posteriors = self.latent_layer(encoder_outputs[:, 0], response_encoder_outputs[:, 0])
            # z_kld_loss, z, priors, posteriors = self.latent_layer(encoder_outputs[:, 0])

        # return encoder_outputs, response_encoder_outputs, z_kld_loss, z, priors, posteriors, src_mask
        return encoder_outputs, None, z_kld_loss, z, priors, posteriors, src_mask