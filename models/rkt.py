from typing import Optional, List, Tuple

import csv
import sys
import torch
import torch.nn as nn

from timm.models.vision_transformer import VisionTransformer, Block
from timm.layers import LayerNorm, get_act_layer, get_norm_layer
from timm.layers.attention import Attention

sys.path.append('/home/ljmarten/ReversibleTokenPruning')
from core.token_manager import TokenManager


def _masked_attn_fill(attn: torch.Tensor, mask: Optional[torch.Tensor], fill_value: float = -1e9):
    """Apply boolean mask efficiently to attention scores.

    - attn: [B, num_heads, N, N]
    - mask: None or [B, 1, N, N] (bool) where True indicates positions to mask
    """
    if mask is None:
        return attn
    if mask.shape != attn.shape:
        mask = mask.expand_as(attn)
    return attn.masked_fill(mask, fill_value)


class AttentionRKT(Attention):
    """Attention que integra TokenManager para pruning/revival.

    Mejoras:
    - usa masked_fill en vez de operaciones float-costosas
    - evita múltiples expansiones innecesarias
    - garantiza un mask de salida consistente
    """
    def __init__(self, token_manager: TokenManager = None, **kwargs):
        super().__init__(**kwargs)
        self.token_manager = token_manager

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape

        # Proyecciones qkv
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q * self.scale

        attn = torch.matmul(q, k.transpose(-2, -1))  # [B, num_heads, N, N]

        # # Aplicar máscara eficientemente
        # # Basándote en attn_mask, calcular con cuántos tokens vivos se calcula la atención
        # if attn_mask is not None:
        #     # attn_mask: [B, 1, N, N] con 0=vivo, !=0=descartado
        #     # Para cada batch, contar cuántos tokens vivos hay (puede variar por batch)
        #     alive_tokens_per_batch = (attn_mask[:, 0, 0, :] == 0).sum(dim=1)  # [B]
        #     print("Tokens usados en el cálculo de la atención:", alive_tokens_per_batch[0].item())
        attn = _masked_attn_fill(attn, attn_mask, fill_value=-1e9)

        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Permitir pruning por TokenManager (implementación interna puede optimizar)
        current_block_attn_mask = None
        if self.token_manager is not None:
            # detach attention scores when passing to token_manager to avoid
            # retaining the large attention graph inside the manager
            current_block_attn_mask = self.token_manager.prune(attn.detach(), attn_mask)

        x_out = torch.matmul(attn, v)
        x_out = x_out.transpose(1, 2).reshape(B, N, C)
        x_out = self.norm(x_out)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)

        if current_block_attn_mask is None:
            if attn_mask is None:
                current_block_attn_mask = torch.zeros((B, 1, N, N), dtype=torch.bool, device=x.device)
            else:
                current_block_attn_mask = attn_mask.to(torch.bool)

        return x_out, current_block_attn_mask


class BlockRKT(Block):
    """Bloque Transformer que usa AttentionRKT.

    Optimiza el forward evitando reasignaciones intermedias innecesarias.
    """
    def __init__(self, token_manager: TokenManager = None, **kwargs):
        raw_norm = kwargs.get('norm_layer')
        raw_act = kwargs.get('act_layer')
        resolved_norm = get_norm_layer(raw_norm) or LayerNorm
        resolved_act = get_act_layer(raw_act) or nn.GELU
        kwargs['norm_layer'] = resolved_norm
        kwargs['act_layer'] = resolved_act

        super().__init__(**kwargs)
        self.token_manager = token_manager

        self.attn = AttentionRKT(
            token_manager=token_manager,
            dim=kwargs.get('dim', getattr(self, 'dim', None)),
            num_heads=kwargs.get('num_heads', getattr(self, 'num_heads', None)),
            qkv_bias=kwargs.get('qkv_bias', getattr(self, 'qkv_bias', True)),
            qk_norm=kwargs.get('qk_norm', getattr(self, 'qk_norm', False)),
            scale_norm=kwargs.get('scale_attn_norm', getattr(self, 'scale_attn_norm', None)),
            proj_bias=kwargs.get('proj_bias', getattr(self, 'proj_bias', True)),
            attn_drop=kwargs.get('attn_drop', getattr(self, 'attn_drop', 0.0)),
            proj_drop=kwargs.get('proj_drop', getattr(self, 'proj_drop', 0.0)),
            norm_layer=resolved_norm,
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Attention block
        residual = x
        x_norm = self.norm1(x)
        x_attn, current_block_attn_mask = self.attn(x_norm, attn_mask)
        x_attn = self.ls1(x_attn)
        x = residual + self.drop_path1(x_attn)

        # MLP block
        residual = x
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        x_mlp = self.ls2(x_mlp)
        x = residual + self.drop_path2(x_mlp)

        return x, current_block_attn_mask


class VisionTransformerRKT(VisionTransformer):
    def __init__(self, *args, token_manager: TokenManager = None, can_tokens_revive: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_manager = token_manager or TokenManager(None, None)
        self.can_tokens_revive = can_tokens_revive

        # Reemplazar bloques por bloques RKT con schedule de drop path basado en self
        len_blocks = len(self.blocks)
        dp_rate = getattr(self, 'drop_path_rate', 0.0)
        drp = [float(x) for x in torch.linspace(0.0, dp_rate, len_blocks)]

        blocks = []
        for i in range(len_blocks):
            blocks.append(
                BlockRKT(
                    token_manager=self.token_manager,
                    dim=getattr(self, 'embed_dim', kwargs.get('embed_dim')),
                    num_heads=getattr(self, 'num_heads', kwargs.get('num_heads')),
                    mlp_ratio=getattr(self, 'mlp_ratio', kwargs.get('mlp_ratio')),
                    qkv_bias=getattr(self, 'qkv_bias', kwargs.get('qkv_bias', True)),
                    qk_norm=getattr(self, 'qk_norm', kwargs.get('qk_norm', False)),
                    scale_attn_norm=getattr(self, 'scale_attn_norm', kwargs.get('scale_attn_norm', None)),
                    scale_mlp_norm=getattr(self, 'scale_mlp_norm', kwargs.get('scale_mlp_norm', None)),
                    proj_bias=getattr(self, 'proj_bias', kwargs.get('proj_bias', True)),
                    init_values=getattr(self, 'init_values', kwargs.get('init_values', None)),
                    proj_drop=getattr(self, 'proj_drop_rate', kwargs.get('proj_drop_rate', 0.0)),
                    attn_drop=getattr(self, 'attn_drop_rate', kwargs.get('attn_drop_rate', 0.0)),
                    drop_path=drp[i],
                    norm_layer=kwargs.get('norm_layer'),
                    act_layer=kwargs.get('act_layer'),
                    mlp_layer=kwargs.get('mlp_layer'),
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.token_manager is not None:
            self.token_manager.mode = "train" if mode else "eval"
        return self

    def forward_features_saving_stats(self, x: torch.Tensor) -> torch.Tensor:
        """Forward que guarda estadísticas mínimas sin pandas y con pocas allocs."""
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        B, N, D = x.shape
        attn_mask = torch.zeros((B, 1, N, N), dtype=torch.bool, device=x.device)
        dead_tokens = torch.full((B, N, D), float('nan'), device=x.device)

        stats_rows: List[Tuple[int, List[int], List[int]]] = []

        for i, blk in enumerate(self.blocks):
            x, current_block_attn_mask = blk(x, attn_mask=attn_mask)

            cur_mask_1d = current_block_attn_mask[:, 0, 0, :].to(torch.bool)
            if cur_mask_1d.any():
                mask_exp = cur_mask_1d.unsqueeze(-1).expand(-1, -1, D)
                # store detached copies of dead tokens to break backward flow
                dead_tokens = torch.where(mask_exp, x.detach(), dead_tokens)

            attn_mask = attn_mask | current_block_attn_mask

            # pass detached tensors to token_manager to avoid building graph there
            revived = self.token_manager.revive(x[:, 0, :].detach(), dead_tokens.detach(), attn_mask) if self.token_manager is not None else torch.zeros((B, N), dtype=torch.bool, device=x.device)

            if revived.any():
                revived_exp = revived.unsqueeze(-1).expand(-1, -1, D)
                # use detached dead_tokens when copying back into x
                x = torch.where(revived_exp, dead_tokens.detach(), x)
                rev_4d = revived.unsqueeze(1).unsqueeze(2).expand(B, 1, N, N)
                attn_mask = attn_mask ^ rev_4d
                dead_tokens = torch.where(revived_exp, torch.full_like(dead_tokens, float('nan')), dead_tokens)

            # record small sample (first batch element)
            attn_mask_sample = attn_mask[0, 0, 0, :].to(torch.uint8).cpu().tolist()
            revived_sample = revived[0].to(torch.uint8).cpu().tolist()
            stats_rows.append((i, attn_mask_sample, revived_sample))

        x = self.norm(x)

        # write compact CSV
        with open('token_stats_3.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['block', 'attn_mask_bits', 'revived_bits'])
            for block_idx, att_bits, rev_bits in stats_rows:
                writer.writerow([block_idx, ''.join(map(str, att_bits)), ''.join(map(str, rev_bits))])

        exit()

    def forward_features(self, x: torch.Tensor, can_tokens_revive: bool) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        B, N, D = x.shape
        attn_mask = torch.zeros((B, 1, N, N), dtype=torch.bool, device=x.device)
        dead_tokens = torch.full((B, N, D), float('nan'), device=x.device)

        for blk in self.blocks:
            x, current_block_attn_mask = blk(x, attn_mask=attn_mask)

            if can_tokens_revive:
                cur_mask_1d = current_block_attn_mask[:, 0, 0, :].to(torch.bool)
                if cur_mask_1d.any():
                    mask_exp = cur_mask_1d.unsqueeze(-1).expand(-1, -1, D)
                    dead_tokens = torch.where(mask_exp, x.detach(), dead_tokens)

            attn_mask = attn_mask | current_block_attn_mask

            if can_tokens_revive and self.token_manager is not None:
                alive_tokens = torch.where(attn_mask[:, 0, 0, :].unsqueeze(-1).expand(-1, -1, D), torch.full_like(x, float('nan')), x)
                revived_tokens = self.token_manager.revive(x[:, 0, :].detach(), dead_tokens.detach(), alive_tokens, attn_mask) # pass detached tensors to avoid retaining gradients through manager
                if revived_tokens.any():
                    revived_exp = revived_tokens.unsqueeze(-1).expand(-1, -1, D)
                    x = torch.where(revived_exp, dead_tokens.detach(), x)
                    rev_4d = revived_tokens.unsqueeze(1).unsqueeze(2).expand(B, 1, N, N)
                    attn_mask = attn_mask ^ rev_4d
                    dead_tokens = torch.where(revived_exp, torch.full_like(dead_tokens, float('nan')), dead_tokens)

        x = self.norm(x)

        # exit()

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x, can_tokens_revive=self.can_tokens_revive)
        x = self.forward_head(x)
        return x
