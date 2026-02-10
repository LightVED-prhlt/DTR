import torch
from core.revival_base import BaseRevivor
import torch.nn.functional as F

class AffinityRevivor(BaseRevivor):
    def __init__(self, criterion: str):
        self.criterion = criterion

    def compute_scores(
        self, 
        cls_emb: torch.Tensor,      # [B, D]
        dead_tokens: torch.Tensor,  # [B, N_dead, D]
        alive_tokens: torch.Tensor, # [B, N_alive, D]
    ) -> torch.Tensor:
        """
        cls_emb: [B, D]
        dead_tokens: [B, N_dead, D]
        alive_tokens: [B, N_alive, D]
        """
        cls_norm = F.normalize(cls_emb, dim=-1)  # [B, D]
        dead_norm = F.normalize(dead_tokens, dim=-1)  # [B, N_dead, D]

        if self.criterion == "C1": # Similaridad coseno
            scores = F.cosine_similarity(cls_norm.unsqueeze(1), dead_norm, dim=-1)  # [B, N_dead]

        elif self.criterion == "C2": # Afinidad con los tokens más relevantes vivos
            alive_norm = F.normalize(alive_tokens, dim=-1)  # [B, N_alive, D]
            sim = torch.einsum("bnd,bmd->bnm", dead_norm, alive_norm)  # [B, N_dead, N_alive]
            scores, _ = sim.max(dim=-1)  # [B, N_dead]
        
        elif self.criterion == "C3": # Reconstruction-based relevance
            proj = torch.einsum("bd,bnd->bn", cls_norm, dead_norm)  # [B, N_dead]
            scores = torch.abs(proj)  # [B, N_dead]

        elif self.criterion == "C4": # L2-norm of token embeddings
            scores = torch.norm(dead_tokens, dim=-1)  # [B, N_dead]

        else:
            raise ValueError(f"Criterio de revival desconocido: {self.criterion}")
        
        return scores