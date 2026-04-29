from typing import Union
from abc import ABC, abstractmethod
import torch

class BasePruner(ABC):
    """Interfaz genérica para estrategias de pruning."""
    @abstractmethod
    def compute_scores(self, attn: torch.Tensor, cls_idx: int = 0) -> torch.Tensor:
        """
        Args:
            attn: tensor [B, H, N, N] con pesos de atención (heads, tokens, tokens)
            cls_idx: índice del token CLS (por defecto 0)
        
        Returns:
            scores: tensor [B, N] con una puntuación por token (más alto = más importante)
        """
        raise NotImplementedError()
    
    def prune_spatial(self, attn: torch.Tensor, attn_mask: torch.Tensor, ratio_or_n_tokens: Union[float, int]) -> torch.Tensor:
        """
        Igual que `prune` pero realiza la selección únicamente sobre los tokens espaciales
        (todos excepto el token CLS en índice 0). El token CLS se mantiene siempre, y no
        se fuerza su score a `inf` — simplemente se excluye de la selección.

        Args:
            - attn: tensor [B, H, N, N] con pesos de atención
            - attn_mask: máscara [B, 1, N, N] (0=vivo, !=0=descartado)
            - ratio_or_n_tokens: float (fracción a mantener) o int (n tokens a mantener)

        Returns:
            - mask: máscara [B, 1, N, N] indicando qué claves descartar (1) o mantener (0).
        """
        B, _, N, _ = attn.shape

        # Pide las puntuaciones por token
        scores_vec = self.compute_scores(attn)  # [B, N]

        # Estado de claves vivas (0=vivo)
        alive_keys = (attn_mask[:, 0, 0, :] == 0)  # [B, N]

        # Definir tokens espaciales: todos menos CLS (índice 0)
        spatial_alive = alive_keys.clone()
        spatial_alive[:, 0] = False

        # Número de tokens espaciales vivos por batch
        avail_spatial = spatial_alive.sum(dim=1)  # [B]

        # Cuántos tokens espaciales mantener por batch
        if isinstance(ratio_or_n_tokens, float):
            desired_keep_spatial = (ratio_or_n_tokens * avail_spatial.float()).round().long()
        else:
            desired_keep_spatial = torch.full_like(avail_spatial, ratio_or_n_tokens)

        # Asegurar rango válido [0, avail_spatial]
        desired_keep_spatial = torch.clamp(desired_keep_spatial, min=0)
        desired_keep_spatial = torch.min(desired_keep_spatial, avail_spatial)

        # print("Tokens espaciales vivos antes pruning:", avail_spatial[0].item(), "\tTokens espaciales a mantener:", desired_keep_spatial[0].item(), end="\t")

        # Preparar scores: tokens no-espaciales o muertos tendrán -inf para no ser seleccionados
        scores_masked = scores_vec.clone()
        scores_masked[~spatial_alive] = -float('inf')

        # Determinar tokens a mantener por batch
        keep_tokens = torch.zeros((B, N), dtype=torch.bool, device=attn.device)
        
        # Asegurar que CLS siempre se mantiene
        cls_idx = 0
        keep_tokens[:, cls_idx] = True

        for i in range(B):
            k = int(desired_keep_spatial[i].item())
            if k <= 0:
                continue
            # limitar k al disponible
            k = min(k, int(avail_spatial[i].item()))
            row = scores_masked[i]
            # seleccionar top-k entre los tokens espaciales (otros valen -inf)
            topk_idx = torch.topk(row, k, largest=True).indices
            keep_tokens[i] = torch.scatter(keep_tokens[i], 0, topk_idx, True)

        # Tokens descartados: los que no están en keep_tokens
        discard_vector = ~keep_tokens
        # Asegurar CLS nunca descartado
        discard_vector[:, cls_idx] = False

        # Expandir a máscara [B, 1, N, N] marcando columnas (claves) descartadas
        mask = discard_vector.unsqueeze(1).unsqueeze(2).expand(B, 1, N, N).contiguous()

        return mask
    