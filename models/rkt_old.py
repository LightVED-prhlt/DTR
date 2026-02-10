import pandas as pd
from typing import Optional

import torch
import torch.nn as nn

from timm.models.vision_transformer import VisionTransformer, Block
from timm.layers import LayerNorm, get_act_layer, get_norm_layer
from timm.layers.attention import Attention

import sys
sys.path.append('/home/ljmarten/ReversibleTokenPruning')
from core.token_manager import TokenManager


def maybe_add_mask(scores: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
    return scores if attn_mask is None else scores + attn_mask

class AttentionRKT(Attention):
    """
    Attention modificado que integra un TokenManager para permitir
    operaciones de pruning (eliminación de tokens) y revival (revivir tokens)
    en fases de inferencia.

    Parámetros:
    - token_manager (TokenManager|None): objeto que expone las interfaces
      `prune(attn)` y `revive(x)`/`revive_tokens`. Si es `None` se comporta
      como una atención estándar.

    Nota: esta clase delega la mayoría de la lógica de proyección y normalización
    a la implementación base de `Attention` (de timm). Aquí se añade la lógica
    para invocar `token_manager` antes/después de la atención.
    """
    def __init__(self, token_manager: TokenManager = None, **kwargs):
        super().__init__(**kwargs)
        self.token_manager = token_manager

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de la atención.

        Input:
        - x: tensor [B, N, C]
        - attn_mask: máscara de atención acumulada [B, N] o None donde 0 indica tokens vivos y 1 o más indica tokens descartados

        Output:
        - tensor [B, N, C] con la misma forma que la entrada.
        """
        B, N, C = x.shape

        # Proyecciones qkv y cálculo estándar de atención
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # qkv: [3, B, num_heads, N, head_dim]
        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # [B, num_heads, N, N]
        attn = maybe_add_mask(attn, attn_mask.float() * -1e9) # attn_mask: [B, 1, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Pruning dinámico usando la interfaz de TokenManager
        if self.token_manager:
            current_block_attn_mask = self.token_manager.prune(attn, attn_mask)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)  # x: [B, N, C]
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, current_block_attn_mask
    
class BlockRKT(Block):
    """
    Bloque Transformer que usa `AttentionRKT` en lugar de la atención estándar.

    Este bloque mantiene la API del bloque base de timm para poder sustituirlo
    en arquitecturas existentes.
    """
    def __init__(self, token_manager: TokenManager = None, **kwargs):
        # Resolver internamente norm_layer y act_layer a partir de los valores
        # pasados en kwargs (pueden ser nombres, None, o ya funciones/clases).
        # Esto evita que la lógica de resolución tenga que vivir en
        # VisionTransformerRKT.
        raw_norm = kwargs.get('norm_layer')
        raw_act = kwargs.get('act_layer')
        resolved_norm = get_norm_layer(raw_norm) or LayerNorm
        resolved_act = get_act_layer(raw_act) or nn.GELU
        kwargs['norm_layer'] = resolved_norm
        kwargs['act_layer'] = resolved_act

        super().__init__(**kwargs)
        self.token_manager = token_manager

        # Reemplazar atención por la versión con TokenManager
        self.attn = AttentionRKT(
            token_manager=token_manager,
            dim=kwargs.get('dim'),
            num_heads=kwargs.get('num_heads'),
            qkv_bias=kwargs.get('qkv_bias'),
            qk_norm=kwargs.get('qk_norm'),
            scale_norm=kwargs.get('scale_attn_norm'),
            proj_bias=kwargs.get('proj_bias'),
            attn_drop=kwargs.get('attn_drop'),
            proj_drop=kwargs.get('proj_drop'),
            norm_layer=resolved_norm,
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward sencillo del bloque: atención seguida de MLP con dos skip connections.
        """
        # Attention + Add & Norm
        x_residual = x
        x = self.norm1(x)
        x, current_block_attn_mask = self.attn(x, attn_mask)
        x = self.ls1(x)
        x = x_residual + self.drop_path1(x)

        # MLP + Add & Norm
        x_residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.ls2(x)
        x = x_residual + self.drop_path2(x)

        return x, current_block_attn_mask

class VisionTransformerRKT(VisionTransformer):
    def __init__(self, *args, token_manager: TokenManager = None, can_tokens_revive: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_manager = token_manager or TokenManager(None, None)
        self.can_tokens_revive = can_tokens_revive

        # Reemplazar bloques por bloques RKT
        len_blocks = len(self.blocks)
        drp = [x.item() for x in torch.linspace(0, kwargs['drop_path_rate'], len_blocks)]
        self.blocks = nn.Sequential(*[
            BlockRKT(
                token_manager=self.token_manager,
                dim=kwargs.get('embed_dim'),
                num_heads=kwargs.get('num_heads'),
                mlp_ratio=kwargs.get('mlp_ratio'),
                qkv_bias=kwargs.get('qkv_bias'),
                qk_norm=kwargs.get('qk_norm'),
                scale_attn_norm=kwargs.get('scale_attn_norm'),
                scale_mlp_norm=kwargs.get('scale_mlp_norm'),
                proj_bias=kwargs.get('proj_bias'),
                init_values=kwargs.get('init_values'),
                proj_drop=kwargs.get('proj_drop_rate'),
                attn_drop=kwargs.get('attn_drop_rate'),
                drop_path=drp[i] if 'drop_path_rate' in kwargs else 0.0,
                norm_layer=kwargs.get('norm_layer'),
                act_layer=kwargs.get('act_layer'),
                mlp_layer=kwargs.get('mlp_layer'),
            ) for i in range(len_blocks)
        ])

    def train(self, mode: bool = True):
        super().train(not mode)
        if self.token_manager is not None:
            self.token_manager.mode = "train" if mode else "eval"
        return self
    
    def forward_features_saving_stats(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature layers (embeddings, transformer blocks, post-transformer norm)."""
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        # Inicializar máscara de atención acumulada
        B, N, D = x.shape
        attn_mask = torch.zeros((B, 1, N, N), dtype=torch.bool, device=x.device) # 0=vivo, 1=descartado

        # Inicializar cementerio de tokens
        dead_tokens = torch.full((B, N, D), float('nan'), device=x.device)

        # Crear un DataFrame que contendrá para cada bloque, una lista de los tokens descartados y revividos
        token_stats = pd.DataFrame(columns=['revived_tokens', 'attn_mask'])

        for i, blk in enumerate(self.blocks):
            x, current_block_attn_mask = blk(x, attn_mask=attn_mask)
            
            # Actualizar cementerio de tokens descartados y preservando su información
            current_block_attn_mask_expanded = current_block_attn_mask[:, 0, 0, :]  # [B, N]
            current_block_attn_mask_expanded = current_block_attn_mask_expanded.unsqueeze(-1).expand(-1, -1, D)
            dead_tokens = torch.where(current_block_attn_mask_expanded, x, dead_tokens)
            
            # Actualizar máscara
            attn_mask = attn_mask | current_block_attn_mask
            token_stats.at[i, 'attn_mask'] = attn_mask[3, 0, 0, :].cpu().numpy().tolist()
            
            # Buscar candidatos a revivir | False -> muertos | True -> vivos
            revived_tokens = self.token_manager.revive(x[:, 0, :], dead_tokens, attn_mask)
            token_stats.at[i, 'revived_tokens'] = revived_tokens[3].cpu().numpy().tolist()
            
            # Incorporar tokens revividos a x
            revived_tokens_expanded = revived_tokens.unsqueeze(-1).expand(-1, -1, D)
            x = torch.where(revived_tokens_expanded, dead_tokens, x)
            
            # Actualizar máscara
            revived_tokens = revived_tokens.unsqueeze(1).unsqueeze(2).expand(B, 1, N, N).contiguous()
            attn_mask = attn_mask ^ (revived_tokens)
            
            # Actualizar cementerio
            dead_tokens = torch.where(revived_tokens_expanded, torch.full_like(dead_tokens, float('nan')), dead_tokens)

        # Guardar DataFrame
        token_stats.to_csv('token_stats_3.csv', index=False)
        exit()

        x = self.norm(x)
        return x
    
    def forward_features(self, x: torch.Tensor, can_tokens_revive: bool) -> torch.Tensor:
        """Forward pass through feature layers (embeddings, transformer blocks, post-transformer norm)."""
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        # Inicializar máscara de atención acumulada
        B, N, D = x.shape
        attn_mask = torch.zeros((B, 1, N, N), dtype=torch.bool, device=x.device) # 0=vivo, 1=descartado

        # Inicializar cementerio de tokens
        dead_tokens = torch.full((B, N, D), float('nan'), device=x.device)

        for blk in self.blocks:
            x, current_block_attn_mask = blk(x, attn_mask=attn_mask)
            
            # Actualizar cementerio de tokens descartados y preservando su información
            if can_tokens_revive:
                current_block_attn_mask_expanded = current_block_attn_mask[:, 0, 0, :]  # [B, N]
                current_block_attn_mask_expanded = current_block_attn_mask_expanded.unsqueeze(-1).expand(-1, -1, D)
                dead_tokens = torch.where(current_block_attn_mask_expanded, x, dead_tokens)
            
            # Actualizar máscara
            attn_mask = attn_mask | current_block_attn_mask
            
            # Buscar candidatos a revivir | False -> muertos | True -> vivos
            if can_tokens_revive:
                # Crear un cielo de tokens vivos temporal que rellena con float('nan') los tokens muertos
                alive_tokens = torch.where(attn_mask[:, 0, 0, :].unsqueeze(-1).expand(-1, -1, D), torch.full_like(x, float('nan')), x)
                revived_tokens = self.token_manager.revive(x[:, 0, :], dead_tokens, alive_tokens, attn_mask)

                # Incorporar tokens revividos a x
                revived_tokens_expanded = revived_tokens.unsqueeze(-1).expand(-1, -1, D)
                x = torch.where(revived_tokens_expanded, dead_tokens, x)
                
                # Actualizar máscara
                revived_tokens = revived_tokens.unsqueeze(1).unsqueeze(2).expand(B, 1, N, N).contiguous()
                attn_mask = attn_mask ^ (revived_tokens)

                # Actualizar cementerio
                dead_tokens = torch.where(revived_tokens_expanded, torch.full_like(dead_tokens, float('nan')), dead_tokens)

        # exit()
        x = self.norm(x)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x, can_tokens_revive=self.can_tokens_revive)
        x = self.forward_head(x)
        return x