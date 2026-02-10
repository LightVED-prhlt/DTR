import torch
import time
import numpy as np
import pandas as pd

# --- Métricas básicas ---
def count_alive_tokens(mask: torch.Tensor):
    """Cuenta el número de tokens vivos en una máscara dada."""
    return mask.sum().item(), mask.numel()

# --- Hooks para FLOPs y activaciones ---
def flops_per_layer(attn_shape, num_tokens_alive):
    """Estimación simple de FLOPs de autoatención."""
    B, H, N, _ = attn_shape
    return (B * H * (N ** 2)) * 2e-6 * (num_tokens_alive / N)

def register_flops_hook(module, flops_dict, name):
    """Hook que registra FLOPs estimados de cada capa."""
    def hook_fn(_, input, output):
        if isinstance(output, torch.Tensor):
            num_tokens = output.shape[1]
            flops_dict[name] = flops_per_layer(output.shape, num_tokens)
    module.register_forward_hook(hook_fn)

# --- Monitoreo de latencia ---
@torch.no_grad()
def measure_latency(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device, warmup: int = 5):
    """Evalúa latencia media por batch pequeño."""
    model.eval().to(device)
    times = []
    for i, (x, _) in enumerate(dataloader):
        x = x.to(device)
        torch.cuda.synchronize()
        t0 = time.time()
        _ = model(x)
        torch.cuda.synchronize()
        if i >= warmup:
            times.append(time.time() - t0)
    return np.mean(times)

# --- Gestión de tokens ---
def summarize_token_stats(stats_dict: list):
    df = pd.DataFrame(stats_dict)

# --- Integración con WandB / Lightning logs ---
def log_wandb_metrics(logger, metrics_dict, step=None):
    """Envía métricas personalizadas a WandB."""
    if logger is not None:
        logger.log_metrics(metrics_dict, step=step)

# --- Utilidades varias ---
def set_seed(seed: int = 42):
    """Establece la semilla para reproducibilidad."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model: torch.nn.Module):
    """Cuenta el número de parámetros entrenables en el modelo."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
