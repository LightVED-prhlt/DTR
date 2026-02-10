import torch
import torch.nn as nn

import timm
from timm.models.vision_transformer import Block
from timm.layers import Mlp, PatchEmbed, get_act_layer, get_norm_layer

from timm_rtk import VisionTransformerRKT

vit_small_cfg = {
    'img_size': (224, 224),
    'patch_size': 16,
    'in_chans': 3,
    'num_classes': 100,
    'global_pool': 'token',
    'embed_dim': 384,
    'depth': 12,
    'num_heads': 6,
    'mlp_ratio': 4.,
    'qkv_bias': True,
    'qk_norm': False,
    'scale_attn_norm': False,
    'scale_mlp_norm': False,
    'proj_bias': True,
    'init_values': None,
    'class_token': True,
    'pos_embed': 'learn',
    'no_embed_class': False,
    'reg_tokens': 0,
    'pre_norm': False,
    'final_norm': True,
    'fc_norm': None,
    'pool_include_prefix': False,
    'dynamic_img_size': False,
    'dynamic_img_pad': False,
    'drop_rate': 0.,
    'pos_drop_rate': 0.,
    'patch_drop_rate': 0.,
    'proj_drop_rate': 0.,
    'attn_drop_rate': 0.,
    'drop_path_rate': 0.,
    'weight_init': '',
    'fix_init': False,
    'embed_layer': PatchEmbed,
    'embed_norm_layer': None,
    'norm_layer': None,
    'act_layer': None,
    'block_fn': Block,
    'mlp_layer': Mlp,
}

class BaseModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool,
        checkpoint_path: str = ""
    ):
        super(BaseModel, self).__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.checkpoint_path = checkpoint_path

        self.__load_model()
    
    def __load_model(self):
        if self.checkpoint_path:
            self.model = VisionTransformerRKT(**vit_small_cfg)
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model."):
                    new_key = k[len("model."):]
                    if new_key.startswith('model.'):
                        new_key = new_key[len('model.'):]
                else:
                    new_key = k
                new_state_dict[new_key] = v
            state_dict = new_state_dict
            self.model.load_state_dict(state_dict)
        else:
            self.model = timm.create_model(self.model_name, pretrained=self.pretrained, num_classes=self.num_classes)