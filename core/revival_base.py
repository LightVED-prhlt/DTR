from typing import Union
from abc import ABC, abstractmethod
import torch

class BaseRevivor(ABC):
    """Interfaz genérica para estrategias de revival de tokens."""
    @abstractmethod
    def compute_scores(self, cls_emb: torch.Tensor, dead_tokens: torch.Tensor, alive_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_emb: tensor [B, D] con la representación del token CLS
            dead_tokens: tensor [B, M, D] con las representaciones de los M tokens eliminados
            alive_tokens: tensor [B, N_alive, D] con las representaciones de los N_alive tokens vivos

        Returns:
            scores: tensor [B, M] con una puntuación por token (más alto = más importante)
        """
        raise NotImplementedError()

    def revive(self, cls_emb: torch.Tensor, dead_tokens: torch.Tensor, alive_tokens: torch.Tensor, attn_mask: torch.Tensor, ratio_or_n_tokens: Union[float, int]) -> torch.Tensor:
        """
        Selecciona los tokens a revivir
        
        Args:
            - cls_emb: tensor [B, D] con la representación del token CLS
            - dead_tokens: tensor [B, N, D] con las representaciones de los M tokens eliminados
            - alive_tokens: tensor [B, N_alive, D] con las representaciones de los N_alive tokens vivos
            - attn_mask: Máscara de atención [B, 1, N, N] donde 0 indica tokens vivos y 1 indica tokens descartados
            - ratio_or_n_tokens: proporción o número de tokens a revivir

        Returns:
            - mask: máscara booleana [B, N] indicando qué tokens revivir
        """
        B, N, _ = dead_tokens.shape

        # Pedir al revivor las puntuaciones por token
        scores = self.compute_scores(cls_emb, dead_tokens, alive_tokens)  # [B, N]

        # attn_mask: [B, 1, N, N] con 0=vivo, !=0=descartado
        # Tomamos la primera fila (queries) y la primera dimensión de relación
        # para obtener qué claves están muertas: attn_mask[:,0,0,j]
        dead_keys = (attn_mask[:, 0, 0, :] != 0)  # [B, N]

        # Número de tokens muertos por batch
        avail = dead_keys.sum(dim=1)  # [B]

        # Cuántos tokens mantener por batch. ratio es la fracción a revivir.
        if isinstance(ratio_or_n_tokens, float):
            desired_revive = (ratio_or_n_tokens * avail.float()).round().long()  # [B]
        else:
            desired_revive = torch.full_like(avail, ratio_or_n_tokens)
        
        desired_revive = torch.clamp(desired_revive, min=0)

        # print("Tokens dead before revival:", avail[0].item(), "\tTokens to revive after pruning:", desired_revive[0].item())
        # print()

        # Preparar scores donde tokens vivos tengan -inf para no ser seleccionados
        scores_masked = scores.clone()
        scores_masked[~dead_keys] = -float('inf')

        # Determinar tokens a mantener por batch
        # 0=no revivir, 1=revivir
        revive_tokens = torch.zeros((B, N), dtype=torch.bool, device=dead_tokens.device)
        for i in range(B):
            k = int(desired_revive[i].item())
            row = scores_masked[i]
            topk_idx = torch.topk(row, k, largest=True).indices
            revive_tokens[i] = torch.scatter(revive_tokens[i], 0, topk_idx, True)

        return revive_tokens
