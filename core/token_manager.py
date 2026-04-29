import torch
from core.pruning_base import BasePruner
from core.revival_base import BaseRevivor

class TokenManager:
    """
    Módulo encargado de aplicar poda (pruning) y resurrección (revival)
    de tokens durante el forward de un Vision Transformer.

    La separación en métodos permite aplicar cada proceso de forma independiente
    (p.ej. sólo pruning en inferencia o ambos durante fine-tuning).
    """

    def __init__(
        self,
        pruner: BasePruner,
        revivor: BaseRevivor,
        prune_ratio_or_n_tokens: float = 0.5,
        revive_ratio_or_n_tokens: float = 0.2,
    ):
        self.pruner = pruner
        self.revivor = revivor
        self.prune_ratio_or_n_tokens = prune_ratio_or_n_tokens
        self.revive_ratio_or_n_tokens = revive_ratio_or_n_tokens

        self.count_for_prune = 0
        self.count_for_revive = 0

        self.stats = []

    # -------------------------------------------------------
    # PRUNING
    # -------------------------------------------------------
    def next_n_tokens_for_prune(self, ratio_or_n_tokens) -> int:
        if isinstance(ratio_or_n_tokens, float):
            return ratio_or_n_tokens
        
        n_token_to_return = ratio_or_n_tokens[self.count_for_prune]
        self.count_for_prune += 1

        if self.count_for_prune >= len(ratio_or_n_tokens):
            self.count_for_prune = 0  # Reset for next call

        return n_token_to_return

    def prune(self, attn: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        Aplica la poda de tokens en base al criterio definido por el pruner.

        Args:
            - attn: Matriz de atención [B, num_heads, N, N]
            - attn_mask: Máscara de atención acumulada [B, 1, N, N] donde 0 indica tokens vivos y 1 indica tokens descartados
            
        Returns:
            - current_block_attn_mask: máscara [B, 1, N, N] indicando qué tokens descartar (1) o mantener (0)
        """
        current_block_attn_mask = self.pruner.prune_spatial(attn, attn_mask, ratio_or_n_tokens=self.next_n_tokens_for_prune(self.prune_ratio_or_n_tokens))

        return current_block_attn_mask

    # -------------------------------------------------------
    # REVIVAL
    # -------------------------------------------------------
    def next_n_tokens_for_revive(self, ratio_or_n_tokens) -> int:
        if isinstance(ratio_or_n_tokens, float):
            return ratio_or_n_tokens
        
        n_token_to_return = ratio_or_n_tokens[self.count_for_revive]
        self.count_for_revive += 1

        if self.count_for_revive >= len(ratio_or_n_tokens):
            self.count_for_revive = 0  # Reset for next call

        return n_token_to_return

    def revive(self, cls_emb: torch.Tensor, dead_tokens: torch.Tensor, alive_tokens: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        Aplica la resurrección de tokens descartados en base al criterio definido por el revivor.

        Args:
            cls_emb: embedding del token CLS [B, D]
            dead_tokens: tokens descartados [B, N, D]
            alive_tokens: tokens vivos [B, M, D]
            attn_mask: máscara de atención acumulada [B, 1, N, N] donde 0 indica tokens vivos y 1 indica tokens descartados
        Returns:
            revived_tokens: mascara booleana [B, N] indicando qué tokens revivir (True) o no (False)
        """
        if self.revive_ratio_or_n_tokens == 0.0:
            batch_size, num_dead_tokens, _ = dead_tokens.size()
            return torch.zeros((batch_size, num_dead_tokens), dtype=torch.bool, device=dead_tokens.device)

        revived_tokens = self.revivor.revive(cls_emb, dead_tokens, alive_tokens, attn_mask, ratio_or_n_tokens=self.next_n_tokens_for_revive(self.revive_ratio_or_n_tokens))

        return revived_tokens        
