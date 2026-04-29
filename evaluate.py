import os
import argparse
import yaml
import logging
import warnings
import time
import numpy as np
import pandas as pd
import polars as pl

from typing import Any, List
from numbers import Number
# from datetime import datetime

from prettytable import PrettyTable
from thop import profile

import torch
import torch.nn as nn

import lightning as L

from torchmetrics.classification import MulticlassAccuracy

from models import load_model
from mydatasets import load_dataset

from pruning.heuristic_pruner import HeuristicPruner
from revival.affinity_revivor import AffinityRevivor
from core.token_manager import TokenManager

# from fvcore.nn import FlopCountAnalysis

# ----------------------------------------
# Configuraciones globales
# ----------------------------------------

os.environ["WANDB_SILENT"] = "true"  # Silenciar mensajes de WandB
os.environ["PYTORCH_LIGHTNING_LOG_LEVEL"] = "ERROR"  # Silenciar mensajes de Lightning

logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("lightning.fabric").setLevel(logging.ERROR)

warnings.filterwarnings("ignore")

torch.set_float32_matmul_precision('medium')

pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

# ----------------------------------------
# Lightning Module principal
# ----------------------------------------
class TokenAdaptationModule(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])

        self.model = model

        self.num_classes = model.num_classes

        # Métricas con torchmetrics
        self.train_acc1 = MulticlassAccuracy(num_classes=self.num_classes, top_k=1)
        self.val_acc1 = MulticlassAccuracy(num_classes=self.num_classes, top_k=1)

        # Top-5 si aplica
        if self.num_classes >= 5:
            self.train_acc5 = MulticlassAccuracy(num_classes=self.num_classes, top_k=5)
            self.val_acc5 = MulticlassAccuracy(num_classes=self.num_classes, top_k=5)
        else:
            self.train_acc5 = self.val_acc5 = None
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)

        # - Actualizar métricas
        self.val_acc1.update(logits, y)
        if self.val_acc5 is not None:
            self.val_acc5.update(logits, y)

        loss = self.criterion(logits, y)
        
        return loss
    
    def on_validation_epoch_end(self):
        acc1 = self.val_acc1.compute()
        self.log("val_acc1", acc1, on_step=False, on_epoch=True)
        self.val_acc1.reset()
        if self.val_acc5:
            acc5 = self.val_acc5.compute()
            self.log("val_acc5", acc5, on_step=False, on_epoch=True)
            self.val_acc5.reset()

# ----------------------------------------
# Parameter Analysis
# ----------------------------------------
def analyze_model_parameters(model):
    """
    Analiza y muestra un desglose de los parámetros de un modelo de PyTorch.

    Args:
        model (torch.nn.Module): El modelo a analizar.
    """
    
    # Diccionario para almacenar los parámetros por bloque o capa principal
    param_counts = {}

    # Itera sobre los parámetros con nombre del modelo
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Agrupa los parámetros por el bloque principal al que pertenecen
        # Por ejemplo, 'blocks.0.mlp.fc1.weight' se agrupará en 'blocks.0'
        primary_key = name.split('.')[0]
        if 'blocks' in primary_key:
            # Agrupa todos los bloques de transformer
            block_num = name.split('.')[1]
            primary_key = f'blocks.{block_num}'

        # Si el nombre tiene más partes, se puede ser más específico
        # Por ejemplo, para desglosar entre atención (attn) y la red MLP
        if 'blocks' in name:
            parts = name.split('.')
            sub_key = f'blocks.{parts[1]}.{parts[2]}' # Ej: 'blocks.0.attn' o 'blocks.0.mlp'
            if sub_key in param_counts:
                param_counts[sub_key] += param.numel()
            else:
                param_counts[sub_key] = param.numel()
        else:
            if primary_key in param_counts:
                param_counts[primary_key] += param.numel()
            else:
                param_counts[primary_key] = param.numel()

    # Crea una tabla para mostrar los resultados de forma ordenada
    table = PrettyTable()
    table.field_names = ["Módulo / Capa", "Número de Parámetros", "% del Total"]
    
    total_params = sum(param_counts.values())
    
    # Ordena los módulos por número de parámetros de forma descendente
    sorted_params = sorted(param_counts.items(), key=lambda item: item[1], reverse=True)
    
    for name, count in sorted_params:
        percentage = (count / total_params) * 100
        table.add_row([name, f"{count:,}", f"{percentage:.2f}%"])
        
    print(f"Análisis de Parámetros para el modelo: {model.__class__.__name__}")
    print(table)
    print(f"Total de parámetros entrenables: {total_params:,}")

# ----------------------------------------
# FLOPs y throughput
# ----------------------------------------
def rfft_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for the rfft/rfftn operator.
    """
    input_shape = inputs[0].type().sizes()
    B, H, W, C = input_shape
    N = H * W
    flops = N * C * np.ceil(np.log2(N))
    return flops

# def get_flops(model, img_size=224, show_details=False, ratios=None):
#     with torch.no_grad():
#         model = model.to('cuda')
#         model.eval()

#         x = torch.randn(1, 3, img_size, img_size).to('cuda')
#         fca1 = FlopCountAnalysis(model, x)
#         flops1 = fca1.total()
#     return flops1 / 1e9

def get_flops_thop(model, img_size=224):
    model = model.to('cuda')
    model.eval()

    x = torch.randn(1, 3, img_size, img_size).to('cuda')
    flops, params = profile(model, inputs=(x,), verbose=False)

    return flops / 1e9

@torch.no_grad()
def get_throughput(images, model, num_warmup=5, num_iters=50):
    model = model.to('cuda')
    model.eval()

    images = images.cuda(non_blocking=True)
    batch_size = images.shape[0]
    for i in range(num_warmup):
        model(images)
    torch.cuda.synchronize()
    print(f"throughput averaged with {num_iters} times")
    tic1 = time.time()
    for i in range(num_iters):
        model(images)
    torch.cuda.synchronize()
    tic2 = time.time()
    # print(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
    # MB = 1024.0 * 1024.0
    # print('memory:', torch.cuda.max_memory_allocated() / MB)

    return num_iters * batch_size / (tic2 - tic1)

# ----------------------------------------
# Token Stats
# ----------------------------------------
def summarize_token_stats(stats_dict: list):
    df = pd.DataFrame(columns=range(1, 13), index=['tokens_alive_before_revival', 'pruned', 'tokens_alive_after_revival', 'revived'])

    for block_idx in range(12):
        df.at['tokens_alive_before_revival', block_idx + 1] = stats_dict[block_idx]['tokens_alive_before_revival']
        df.at['pruned', block_idx + 1] = stats_dict[block_idx]['pruned']
        df.at['tokens_alive_after_revival', block_idx + 1] = stats_dict[block_idx]['tokens_alive_after_revival']
        df.at['revived', block_idx + 1] = stats_dict[block_idx]['revived']

    tokens_alive_before_revival_avg = int(df.loc['tokens_alive_before_revival'].mean())
    pruned_avg = int(df.loc['pruned'].mean())
    tokens_alive_after_revival_avg = int(df.loc['tokens_alive_after_revival'].mean())
    revived_avg = int(df.loc['revived'].mean())

    df_avg = pd.DataFrame({
        'Average': [tokens_alive_before_revival_avg, pruned_avg, tokens_alive_after_revival_avg, revived_avg]
    }, index=['tokens_alive_before_revival', 'pruned', 'tokens_alive_after_revival', 'revived'])

    tokens_alive_before_revival_percent = int((tokens_alive_before_revival_avg / 197) * 100.)
    pruned_percent = int((pruned_avg / 197) * 100.)
    tokens_alive_after_revival_percent = int((tokens_alive_after_revival_avg / 197) * 100.)
    revived_percent = int((revived_avg / 197) * 100.)

    df_avg['Average (%)'] = [
        tokens_alive_before_revival_percent,
        pruned_percent,
        tokens_alive_after_revival_percent,
        revived_percent
    ]

    df_final = pd.concat([df, df_avg], axis=1)

    return df_final
    
# ----------------------------------------
# Main function
# ----------------------------------------
def main(config, calc_flops: bool = False, calc_throughput: bool = False, calc_token_stats: bool = False):
    # --- Seed
    L.seed_everything(config.get('seed', 42), workers=True, verbose=False)
    print(f"🌱 Seed fijada en {config.get('seed', 42)}")

    # Map YAML sections to local variables
    model_cfg = config.get('model', {})
    run_cfg = config.get('run', {})
    pruning_cfg = config.get('pruning', {})
    revival_cfg = config.get('revival', {})
    dataset_cfg = config.get('dataset', {})

    # --- Dataset
    datamodule = load_dataset(
        dataset_cfg.get('name'),
        dataset_cfg.get('data_dir'),
        run_cfg.get('batch_size'),
        model_cfg.get('image_size'),
    )

    # --- Pruner & Revivor
    pruner = HeuristicPruner(criterion=pruning_cfg.get('pruning_criterion'))
    revivor = AffinityRevivor(criterion=revival_cfg.get('revival_criterion'))
    token_manager = TokenManager(
        pruner=pruner,
        revivor=revivor,
        prune_ratio_or_n_tokens=pruning_cfg.get('prune_ratio'),
        revive_ratio_or_n_tokens=revival_cfg.get('revive_ratio'),
    )

    # --- Modelo
    model = load_model(
        model_name=model_cfg.get('name'),
        num_classes=datamodule.num_classes,
        pretrained=model_cfg.get('pretrained', False),
        checkpoint_path=model_cfg.get('checkpoint_path', None),
        token_manager=token_manager,
        can_tokens_revive=run_cfg.get('can_tokens_revive', False),
    )

    # n_params = sum(p.numel() for p in model.parameters())
    # print(f"\n🔥 Número de parámetros del modelo: {n_params / 1e6:.1f} M\n")
    # # analyze_model_parameters(model)

    # --- Lightning Module
    lightning_module = TokenAdaptationModule(
        model=model,
    )

    # --- Trainer
    trainer = L.Trainer(
        accelerator='auto',
        devices='auto',
    )

    # --- Evaluate
    print("\n✅ Evaluando modelo final...")
    val_acc1 = trainer.validate(lightning_module, datamodule=datamodule, verbose=False)[0]['val_acc1']

    # --- FLOPs
    if calc_flops:
        print("\n✅ Calculando FLOPs...")
        flops = get_flops_thop(model, img_size=config['image_size'])
        print(f"🔥 FLOPs: {flops:.2f} GFLOPs")
    
    # --- Throughput
    if calc_throughput:
        print("\n✅ Calculando throughput...")
        images = torch.randn(32, 3, config['image_size'], config['image_size'])
        throughput = get_throughput(images, model)
        print(f"🔥 Throughput: {throughput:.2f} images/s")

    # --- Token stats
    if calc_token_stats:
        from tabulate import tabulate
        print("\n✅ Resumiendo estadísticas de tokens...")
        df_token_stats = summarize_token_stats(token_manager.stats[:12])
        print(tabulate(df_token_stats, headers='keys', tablefmt='pretty'))

    return val_acc1

# ----------------------------------------
# CLI
# ----------------------------------------
TOKENS_PER_BLOCK = {
    0.5: {
        "stepwise": [196,196,196,98,98,98,49,49,49,25,25,25],
        "linear": [196,180,165,149,134,118,103,87,72,56,41,25],
        "exponential": [196,163,135,112,93,77,64,53,44,36,30,25],
    },
    0.6: {
        "stepwise": [196,196,196,118,118,118,71,71,71,43,43,43],
        "linear": [196,182,168,154,140,126,113,99,85,71,57,43],
        "exponential": [196,171,149,130,113,98,86,75,65,57,49,43],
    },
    0.7: {
        "stepwise": [196,196,196,137,137,137,96,96,96,67,67,67],
        "linear": [196,184,173,161,149,137,126,114,102,90,79,67],
        "exponential": [196,178,161,146,133,120,109,99,90,81,74,67],
    }
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TokenAdaptation from YAML config")
    parser.add_argument("--config", "-c", default="eval_config.yaml", help="Ruta al archivo YAML de configuración")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        mode = config['run']['mode']

    if mode == 'inference only':
        acc1 = main(config, calc_flops=False, calc_throughput=False, calc_token_stats=False)
        print(f"\n🔥 Accuracy Top-1: {acc1 * 100:.3f}%\n")

    elif mode == 'selection policy evaluation':
        pruning_schedule = 'stepwise'
        budget_target = 0.7
        tokens_per_block_scheme = TOKENS_PER_BLOCK[budget_target][pruning_schedule]
        best_hold_percent, best_acc1 = None, 0.
        for hold_percent in range(75, 100, 1):
            config['pruning']['prune_ratio'] = [round(tokens * hold_percent / 100.) for tokens in tokens_per_block_scheme]
            config['revival']['revive_ratio'] = [t1 - t2 for t1, t2 in zip(tokens_per_block_scheme, config['pruning']['prune_ratio'])]
            acc1 = main(config)
            print(f"\n🔥 Accuracy Top-1: {acc1 * 100:.3f}% | Hold %: {hold_percent}\n")

            if acc1 > best_acc1:
                best_acc1 = acc1
                best_hold_percent = hold_percent

        print(f"\n🎯 Best Top-1 Acc for {pruning_schedule} at {budget_target} pruning: {best_acc1 * 100:.3f}% | Best Hold %: {best_hold_percent}\n")
        
    elif mode == 'selection policy grid search':
        for budget_target in TOKENS_PER_BLOCK.keys():
            for pruning_schedule in TOKENS_PER_BLOCK[budget_target].keys():
                tokens_per_block_scheme = TOKENS_PER_BLOCK[budget_target][pruning_schedule]
                best_hold_percent, best_acc1 = None, 0.
                for hold_percent in range(75, 100, 1):
                    config['pruning']['prune_ratio'] = [round(tokens * hold_percent / 100.) for tokens in tokens_per_block_scheme]
                    config['revival']['revive_ratio'] = [t1 - t2 for t1, t2 in zip(tokens_per_block_scheme, config['pruning']['prune_ratio'])]
                    acc1 = main(config)
                    print(f"\n🔥 Accuracy Top-1: {acc1 * 100:.3f}% | Hold %: {hold_percent}\n")

                    if acc1 > best_acc1:
                        best_acc1 = acc1
                        best_hold_percent = hold_percent

                print(f"\n🎯 Best Top-1 Acc for {pruning_schedule} at {budget_target} pruning: {best_acc1 * 100:.3f}% | Best Hold %: {best_hold_percent}\n")

    elif mode == 'pruning heuristic grid search':
        table = pl.DataFrame(schema={"Budget Target": pl.Float64, "Pruning Criterion": pl.String, "Top-1 Accuracy": pl.Float64})
        config['run']['can_tokens_revive'] = False
        pruning_schedule = 'stepwise'
        for budget_target in TOKENS_PER_BLOCK:
            tokens_per_block_scheme = TOKENS_PER_BLOCK[budget_target][pruning_schedule]
            for pruning_criterion in ['C1', 'C2', 'C3', 'C4']:
                config['pruning']['pruning_criterion'] = pruning_criterion
                config['pruning']['prune_ratio'] = tokens_per_block_scheme
                acc1 = main(config)
                print(f"\n🔥 Budget Target: {budget_target} | Pruning Criterion: {pruning_criterion} | Accuracy Top-1: {acc1 * 100:.3f}%\n")
                table = pl.concat([table,
                    pl.DataFrame({
                    "Budget Target": [budget_target],
                    "Pruning Criterion": [pruning_criterion],
                    "Top-1 Accuracy": [acc1 * 100.]
                })
                ], how="vertical")

        print("\n📊 Resultados del Grid Search de Heurísticas de Poda:")
        print(table)

    elif mode == 'revival heuristic grid search':
        table = pl.DataFrame(schema={"Budget Target": pl.Float64, "Revival Criterion": pl.String, "Top-1 Accuracy": pl.Float64})
        config['run']['can_tokens_revive'] = True
        config['pruning']['pruning_criterion'] = 'C2'
        pruning_schedule = 'stepwise'
        for budget_target in TOKENS_PER_BLOCK:
            tokens_per_block_scheme = TOKENS_PER_BLOCK[budget_target][pruning_schedule]
            for revival_criterion in ['C1', 'C2', 'C3', 'C4']:
                config['revival']['revival_criterion'] = revival_criterion
                config['pruning']['prune_ratio'] = [round(tokens * 0.8) for tokens in tokens_per_block_scheme]
                config['revival']['revive_ratio'] = [t1 - t2 for t1, t2 in zip(tokens_per_block_scheme, config['pruning']['prune_ratio'])]
                acc1 = main(config)
                print(f"\n🔥 Budget Target: {budget_target} | Revival Criterion: {revival_criterion} | Accuracy Top-1: {acc1 * 100:.3f}%\n")
                table = pl.concat([table,
                    pl.DataFrame({
                    "Budget Target": [budget_target],
                    "Revival Criterion": [revival_criterion],
                    "Top-1 Accuracy": [acc1 * 100.]
                })
                ], how="vertical")

        print("\n📊 Resultados del Grid Search de Heurísticas de Revivificación:")
        print(table)
    else:
        raise ValueError(f"Modo desconocido: {mode}")
    