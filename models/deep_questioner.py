from torch import nn
import torch
from torch.functional import Tensor
from models.transformer_layers import Encoder, generate_pad_mask

class DeepQuestioner(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args=args
        self.encoder = Encoder(None, args.hidden_dim, num_layers=args.num_layers, num_heads=args.num_heads, 
                                total_key_depth=args.hidden_dim, total_value_depth=args.hidden_dim,
                                filter_size=args.pwffn_dim)

    def forward(self, text_encoding, image_feature_maps, src_mask):
        cat_features = torch.cat((image_feature_maps, text_encoding), dim=1)
        shallow_alignment = self.encoder(cat_features, None)
        shallow_cat_features = torch.cat((image_feature_maps, shallow_alignment), dim=1)
        deep_alignment = self.encoder(shallow_cat_features, None)
        return deep_alignment