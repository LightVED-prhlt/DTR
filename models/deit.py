from typing import Optional

import torch
import torch.nn as nn

import timm
from timm.models.vision_transformer import Block
from timm.layers import Mlp, PatchEmbed
from timm.models import register_model

from .rkt import VisionTransformerRKT

import sys
sys.path.append('/home/ljmarten/ReversibleTokenPruning')
from core.token_manager import TokenManager

deit_tiny_cfg = {
    'img_size': (224, 224),
    'patch_size': 16,
    'in_chans': 3,
    'global_pool': 'token',
    'embed_dim': 192,
    'depth': 12,
    'num_heads': 3,
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

deit_small_cfg = {
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

deit_base_cfg = {
    'img_size': (224, 224),
    'patch_size': 16,
    'in_chans': 3,
    'global_pool': 'token',
    'embed_dim': 768,
    'depth': 12,
    'num_heads': 12,
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

class DeiTRKT(nn.Module):
    MODEL_CONFIGS = {
        'tiny': {
            'cfg': deit_tiny_cfg,
            'timm_name': 'deit_tiny_patch16_224.fb_in1k'
        },
        'small': {
            'cfg': deit_small_cfg,
            'timm_name': 'deit_small_patch16_224.fb_in1k'
        },
        'base': {
            'cfg': deit_base_cfg,
            'timm_name': 'deit_base_patch16_224.fb_in1k'
        }
    }

    def __init__(
        self,
        model_size: str,
        num_classes: int = 1000,
        pretrained: bool = False,
        checkpoint_path: str = "",
        token_manager: Optional[TokenManager] = None,
        can_tokens_revive: bool = False
    ):
        """
        Args:
            model_size (str): El tamaño del modelo DeiT. Debe ser 'tiny', 'small' o 'base'.
            num_classes (int): Número de clases de salida.
            pretrained (bool): Si es True, carga los pesos pre-entrenados de timm (ignorado si se proporciona checkpoint_path).
            checkpoint_path (str): Ruta a un checkpoint local para cargar los pesos.
            token_manager (Optional[TokenManager]): Gestor de tokens para la arquitectura RKT.
            can_tokens_revive (bool): Flag para la arquitectura RKT.
        """
        super(DeiTRKT, self).__init__()

        if model_size not in self.MODEL_CONFIGS:
            raise ValueError(f"El tamaño del modelo '{model_size}' no es válido. Opciones: {list(self.MODEL_CONFIGS.keys())}")
        
        # Seleccionar configuración del modelo
        config = self.MODEL_CONFIGS[model_size]
        model_cfg = config['cfg'].copy()
        timm_model_name = config['timm_name']

        # Actualizar configuración y crear modelo
        model_cfg['num_classes'] = num_classes
        self.model = VisionTransformerRKT(token_manager=token_manager, can_tokens_revive=can_tokens_revive, **model_cfg)

        # Guardar atributos
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.token_manager = token_manager
        self.checkpoint_path = checkpoint_path
        self.can_tokens_revive = can_tokens_revive

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
            print(f"Pesos cargados desde el checkpoint: {self.checkpoint_path}")
        else:
            timm_model = timm.create_model(timm_model_name, pretrained=self.pretrained, num_classes=self.num_classes)
            self.model.load_state_dict(timm_model.state_dict(), strict=True)
            print(f"Pesos pre-entrenados '{timm_model_name}' cargados desde timm")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward de la envoltura que delega en ``.

        Recibe `x` con forma [B, 3, H, W] y devuelve logits [B, num_classes]
        (o la forma que proporcione el backbone).
        """
        x = self.model(x)
        return x


# Registered factories for timm
# @register_model
# def deit_tiny_rkt(pretrained: bool = False, num_classes: int = 1000, **kwargs) -> nn.Module:
#     return DeiTRKT('tiny', num_classes=num_classes, pretrained=pretrained, **kwargs)


# @register_model
# def deit_small_rkt(pretrained: bool = False, num_classes: int = 1000, **kwargs) -> nn.Module:
#     return DeiTRKT('small', num_classes=num_classes, pretrained=pretrained, **kwargs)


# @register_model
# def deit_base_rkt(pretrained: bool = False, num_classes: int = 1000, **kwargs) -> nn.Module:
#     return DeiTRKT('base', num_classes=num_classes, pretrained=pretrained, **kwargs)
    