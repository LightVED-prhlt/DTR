from typing import Optional

import torch
import torch.nn as nn

import timm
from timm.models.vision_transformer import Block
from timm.layers import Mlp, PatchEmbed
from timm.models import register_model

from .rkt import VisionTransformerRKT
from core.token_manager import TokenManager

vit_small_cfg = {
    'img_size': (224, 224),
    'patch_size': 16,
    'in_chans': 3,
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
    
class ViTSmallRKT(nn.Module):
    def __init__(
        self,
        num_classes: int = 100,
        pretrained: bool = False,
        checkpoint_path: str = "",
        token_manager: Optional[TokenManager] = None,
        can_tokens_revive: bool = False
    ):
        super(ViTSmallRKT, self).__init__()

        vit_small_cfg['num_classes'] = num_classes
        self.model = VisionTransformerRKT(token_manager=token_manager, can_tokens_revive=can_tokens_revive, **vit_small_cfg)
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.checkpoint_path = checkpoint_path

        # Inicializar state_dict por defecto para evitar NameError si no se carga
        state_dict = None
        if self.checkpoint_path:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)

            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_key = k[len('model.'):]
                    if new_key.startswith('model.'):
                        new_key = new_key[len('model.'):]
                else:
                    new_key = k
                new_state_dict[new_key] = v
            self.model.load_state_dict(new_state_dict)
        else:
            self.timm_model = timm.create_model('vit_small_patch16_224.augreg_in1k', pretrained=self.pretrained, num_classes=self.num_classes)
            self.model.load_state_dict(self.timm_model.state_dict(), strict=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward de la envoltura que delega en `VisionTransformerRKT`.

        Recibe `x` con forma [B, 3, H, W] y devuelve logits [B, num_classes]
        (o la forma que proporcione el backbone).
        """
        x = self.model(x)
        return x


@register_model
def vit_small_rkt(pretrained: bool = False, num_classes: int = 1000, **kwargs) -> nn.Module:
    return ViTSmallRKT(num_classes=num_classes, pretrained=pretrained, **kwargs)
        