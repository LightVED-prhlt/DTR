import torch
from core.pruning_base import BasePruner

class HeuristicPruner(BasePruner):
    def __init__(self, criterion: str):
        self.criterion = criterion

    def compute_scores(self, attn: torch.Tensor, cls_idx: int = 0) -> torch.Tensor:
        """
        Args:
            attn: tensor [B, H, N, N] con pesos de atención (heads, tokens, tokens)
            cls_idx: índice del token CLS (por defecto 0)
        """
        attn_mean = attn.mean(dim=1)  # [B, N, N]

        if self.criterion == "C1": # Atención hacia CLS
            scores = attn_mean[:, cls_idx, :]  # [B, N]

        elif self.criterion == "C2": # Contribución entrante
            scores = attn_mean.sum(dim=1)  # [B, N]

        elif self.criterion == "C3": # Contribución saliente
            scores = attn_mean.sum(dim=2)  # [B, N]

        elif self.criterion == "C4": # Entropía saliente
            p = attn_mean / (attn_mean.sum(dim=-1, keepdim=True) + 1e-6)  # [B, N, N]
            entropy = -(p * p.log()).sum(dim=-1)  # [B, N]
            scores = -entropy  # [B, N]

        elif self.criterion == "C5": # Entrante + Saliente
            incoming = attn_mean.sum(dim=1)  # [B, N]
            outgoing = attn_mean.sum(dim=2)  # [B, N]
            scores = incoming + outgoing  # [B, N]

        else:
            raise ValueError(f"Criterio desconocido: {self.criterion}")

        return scores