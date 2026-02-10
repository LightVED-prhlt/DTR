from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer, Block
from timm.layers import Mlp, PatchEmbed, LayerNorm, get_act_layer, get_norm_layer
from timm.layers.attention import Attention, maybe_add_mask

from token_manager import TokenManager

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

class AttentionRKT(Attention):
    def __init__(self, token_manager: TokenManager = None, **kwargs):
        super().__init__(**kwargs)
        self.token_manager = token_manager

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # Revival antes de atención (solo en inferencia)
        if not self.training and self.token_manager is not None:
            if hasattr(self.token_manager, 'revive_tokens'):
                x = self.token_manager.revive(x)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # qkv: [3, B, num_heads, N, head_dim]
        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # [B, num_heads, N, N]
        # attn = maybe_add_mask(attn, attn_mask)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Pruning dinámico usando la interfaz de TokenManager (prune/revive)
        if self.token_manager is not None:
            # TokenManager.prune espera atención [B, num_heads, N, N]
            try:
                attn, mask_keep = self.token_manager.prune(attn)
                # mask_keep: [B, N] -> aplicarlo a x para reducir tokens si es necesario
                # conservar token CLS (asumimos idx 0)
                keep_idxs = mask_keep
                # Si hay hard pruning, reducir x para la siguiente operación
                if not self.training and keep_idxs is not None:
                    # construir nuevos tokens por batch
                    new_x = []
                    for b in range(B):
                        kept = keep_idxs[b].nonzero(as_tuple=False).squeeze(-1)
                        new_x.append(x[b, kept, :])
                    # pad o mantener lista; aquí asignamos tensor concatenado variable por batch
                    # para simplificar, almacenamos x como lista para recálculo de qkv más abajo
                    x_reduced = torch.nn.utils.rnn.pad_sequence(new_x, batch_first=True)
                else:
                    x_reduced = x
            except Exception:
                x_reduced = x

        # Recalcular atención si usamos una versión reducida de x (hard pruning)
        # En la implementación actual x_reduced puede tener padding; para mantener
        # compatibilidad simple, si no hay reducción usamos x original.
        try:
            x_use = x_reduced
        except NameError:
            x_use = x

        if not self.training and x_use.shape[1] != N:
            qkv = self.qkv(x_use).reshape(B, x_use.shape[1], 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)   # qkv: [3, B, num_heads, N', head_dim]
            q, k = self.q_norm(q), self.k_norm(k)
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # [B, num_heads, N', N']
            attn = attn.softmax(dim=-1)

        # Ajustar producto attn @ v: usar dimensiones actuales de attn y v
        xv = (attn @ v).transpose(1, 2)  # [B, N_use, C]
        # Si hubo reducción de tokens, xv puede tener N' != N. Para mantener la
        # forma original rellenamos con ceros hasta N si es necesario.
        if xv.shape[1] != N:
            padded = x.new_zeros(B, N, C)
            padded[:, :xv.shape[1], :] = xv
            xv = padded
        x = xv.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    
class BlockRKT(Block):
    def __init__(self, token_manager: TokenManager = None, **kwargs):
        super().__init__(**kwargs)
        self.token_manager = token_manager

        # Reemplazar atención
        self.attn = AttentionRKT(
            token_manager=token_manager,
            dim=kwargs['dim'],
            num_heads=kwargs['num_heads'],
            qkv_bias=kwargs['qkv_bias'],
            qk_norm=kwargs['qk_norm'],
            scale_norm=kwargs['scale_attn_norm'],
            proj_bias=kwargs['proj_bias'],
            attn_drop=kwargs['attn_drop'],
            proj_drop=kwargs['proj_drop'],
            norm_layer=kwargs['norm_layer'],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))

        return x
    
class VisionTransformerRKT(VisionTransformer):
    def __init__(self, *args, token_manager: TokenManager = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_manager = token_manager or TokenManager(None, None)

        # Reemplazar bloques por bloques RKT
        len_blocks = len(self.blocks)
        drp = [x.item() for x in torch.linspace(0, kwargs['drop_path_rate'], len_blocks)]
        self.blocks = nn.Sequential(*[
            BlockRKT(
                token_manager=self.token_manager,
                dim=kwargs['embed_dim'],
                num_heads=kwargs['num_heads'],
                mlp_ratio=kwargs['mlp_ratio'],
                qkv_bias=kwargs['qkv_bias'],
                qk_norm=kwargs['qk_norm'],
                scale_attn_norm=kwargs['scale_attn_norm'],
                scale_mlp_norm=kwargs['scale_mlp_norm'],
                proj_bias=kwargs['proj_bias'],
                init_values=kwargs['init_values'],
                proj_drop=kwargs['proj_drop_rate'],
                attn_drop=kwargs['attn_drop_rate'],
                drop_path=drp[i] if 'drop_path_rate' in kwargs else 0.0,
                norm_layer=get_norm_layer(kwargs['norm_layer']) or LayerNorm,
                act_layer=get_act_layer(kwargs['act_layer']) or nn.GELU,
                mlp_layer=kwargs['mlp_layer'],
            ) for i in range(len_blocks)
        ])

    def train(self, mode: bool = True):
        super().train(mode)
        if self.token_manager is not None:
            self.token_manager.mode = "train" if mode else "eval"
        return self
    
def create_model(checkpoint_path: str = "", token_manager: Optional[TokenManager] = None, **kwargs) -> VisionTransformerRKT:
    # from timm import create_model as timm_create_model
    # base_model = timm_create_model('vit_small_patch16_224')
    
    # Crear modelo RKT
    model_rkt = VisionTransformerRKT(
        token_manager=token_manager,
        **vit_small_cfg
    )

    # Copiar pesos del modelo base
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
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
    model_rkt.load_state_dict(state_dict)

    return model_rkt
    
if __name__ == "__main__":
    model = create_model(checkpoint_path="/home/ljmarten/ReversibleTokenPruning/checkpoints/vit-cifar100-epoch271-val_acc0.8048.ckpt")
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(out.shape)
        