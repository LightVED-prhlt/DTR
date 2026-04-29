import os
import argparse
import yaml
import logging
import warnings
import numpy as np

from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from timm import create_model
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from torchmetrics.classification import MulticlassAccuracy

from models import load_model
from mydatasets import load_dataset

from pruning.heuristic_pruner import HeuristicPruner
from revival.affinity_revivor import AffinityRevivor
from core.token_manager import TokenManager

# ----------------------------------------
# Configuraciones globales
# ----------------------------------------
os.environ["WANDB_SILENT"] = "true"  # Silenciar mensajes de WandB
os.environ["PYTORCH_LIGHTNING_LOG_LEVEL"] = "ERROR"  # Silenciar mensajes de Lightning

logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("lightning.fabric").setLevel(logging.ERROR)

warnings.filterwarnings("ignore")

torch.set_float32_matmul_precision('medium')

# ----------------------------------------
# Lightning Module principal
# ----------------------------------------
class TokenAdaptationModule(L.LightningModule):
    def __init__(self, model, lr, mixup_config=None, teacher=None, distill_config=None):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'teacher'])

        self.model = model
        self.teacher = teacher

        if self.teacher is not None:
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

        self.num_classes = model.num_classes

        # Métricas con torchmetrics
        self.train_acc1 = MulticlassAccuracy(num_classes=self.num_classes, top_k=1)
        self.val_acc1 = MulticlassAccuracy(num_classes=self.num_classes, top_k=1)
        self.test_acc1 = MulticlassAccuracy(num_classes=self.num_classes, top_k=1)

        # Top-5 si aplica
        if self.num_classes >= 5:
            self.train_acc5 = MulticlassAccuracy(num_classes=self.num_classes, top_k=5)
            self.val_acc5 = MulticlassAccuracy(num_classes=self.num_classes, top_k=5)
            self.test_acc5 = MulticlassAccuracy(num_classes=self.num_classes, top_k=5)
        else:
            self.train_acc5 = self.val_acc5 = self.test_acc5 = None

        # Configurar Mixup y CutMix
        self.mixup_config = mixup_config or {}
        self.mixup_active = (
            self.mixup_config.get("mixup_alpha", 0.0) > 0.0
            or self.mixup_config.get("cutmix_alpha", 0.0) > 0.0
            or self.mixup_config.get("cutmix_minmax", None) is not None
        )
        if self.mixup_active:
            print("⚡ Mixup/CutMix activado\n")
            self.mixup_fn = Mixup(
                mixup_alpha=self.mixup_config.get("mixup_alpha", 0.0),
                cutmix_alpha=self.mixup_config.get("cutmix_alpha", 0.0),
                cutmix_minmax=self.mixup_config.get("cutmix_minmax", None),
                prob=self.mixup_config.get("prob", 1.0),
                switch_prob=self.mixup_config.get("switch_prob", 0.5),
                mode=self.mixup_config.get("mode", "batch"),
                label_smoothing=self.mixup_config.get("label_smoothing", 0.0),
                num_classes=self.num_classes,
            )
        else:
            self.mixup_fn = None

        # Configurar función de pérdida
        if self.mixup_fn is not None:
            self.criterion_train = SoftTargetCrossEntropy()
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.mixup_config["label_smoothing"])

    def _step(self, batch, stage: str):
        x, y = batch

        # Aplicar Mixup
        if stage == "train" and self.mixup_active:
            x, y = self.mixup_fn(x, y)

        logits = self.model(x)

        if stage == "train" and self.mixup_active:
            loss = self.criterion_train(logits, y)
        else:
            loss = self.criterion(logits, y)

        # --- Distillation modes: 'none', 'soft' (KD/attention), 'hard' (teacher labels)
        distill_mode = self.hparams.distill_config["mode"]
        distill_weight = self.hparams.distill_config["weight"]
        if self.teacher is not None and stage == "train" and distill_mode != 'none' and distill_weight > 0.0:
            try:
                if distill_mode == 'soft':
                        with torch.no_grad():
                            t_logits = self.teacher(x)
                        T = self.hparams.distill_config["temp"]
                        p_student = F.log_softmax(logits / T, dim=1)
                        p_teacher = F.softmax(t_logits / T, dim=1)
                        distill_loss = F.kl_div(p_student, p_teacher, reduction='batchmean') * (T * T)
                elif distill_mode == 'hard':
                    # Use teacher hard labels (argmax) as targets
                    with torch.no_grad():
                        t_logits = self.teacher(x)
                        t_labels = t_logits.argmax(dim=1)
                    distill_loss = F.cross_entropy(logits, t_labels)
                else:
                    distill_loss = 0.0
            except Exception:
                distill_loss = 0.0

            loss = loss + distill_weight * distill_loss

        if y.dim() > 1 and y.size(1) == self.num_classes:
            y_for_metrics = y.argmax(dim=1)
        else:
            y_for_metrics = y

        return loss, logits, y_for_metrics
        
    def training_step(self, batch, batch_idx):
        loss, logits, y_for_metrics = self._step(batch, "train")

        # - Actualizar métricas
        self.train_acc1.update(logits, y_for_metrics)
        if self.train_acc5:
            self.train_acc5.update(logits, y_for_metrics)

        # - Loguear pérdida
        self.log("train/loss", loss, on_step=True, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, logits, y_for_metrics = self._step(batch, "val")

        # - Actualizar métricas
        self.val_acc1.update(logits, y_for_metrics)
        if self.val_acc5:
            self.val_acc5.update(logits, y_for_metrics)

        # - Loguear pérdida
        self.log("val/loss", loss, on_step=False, on_epoch=True)

        return loss       
    
    def configure_optimizers(self):
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue  # Saltar parámetros congelados
            if "bias" in name or "norm" in name or "bn" in name:
                no_decay.append(param)
            else:
                decay.append(param)

        param_groups = [
            {"params": decay, "weight_decay": 5e-2},
            {"params": no_decay, "weight_decay": 0.0},
        ]

        lr = self.hparams.lr
        optimizer = torch.optim.AdamW(param_groups, lr=lr)

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs * self.trainer.estimated_stepping_batches // self.trainer.accumulate_grad_batches,
            eta_min=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Actualización por batch
                "frequency": 1,
            }
        }
    
    def on_train_epoch_end(self):
        acc1 = self.train_acc1.compute()
        self.log("train/acc1", acc1, on_step=False, on_epoch=True)
        self.train_acc1.reset()
        if self.train_acc5:
            acc5 = self.train_acc5.compute()
            self.log("train/acc5", acc5, on_step=False, on_epoch=True)
            self.train_acc5.reset()

    def on_validation_epoch_end(self):
        acc1 = self.val_acc1.compute()
        self.log("val/acc1", acc1, on_step=False, on_epoch=True) # wandb
        self.log("val_acc1", acc1, on_step=False, on_epoch=True, logger=False) # checkpoint callback
        self.val_acc1.reset()
        if self.val_acc5:
            acc5 = self.val_acc5.compute()
            self.log("val/acc5", acc5, on_step=False, on_epoch=True)
            self.val_acc5.reset()

# ----------------------------------------
# Aux functions
# ----------------------------------------
def generate_run_name(config):
    # Extraer información de las secciones
    model_name = config.get('model', {}).get('name', 'unknown-model')
    simulate_revival = config.get('run', {}).get('simulate_revival', False)
    distill_mode = config.get('run', {}).get('distill', {}).get('mode', 'none')
    lr = config.get('run', {}).get('lr', 0)
    
    # Extraer algo específico de tu proyecto (ej: ratio de pruning)
    prune_ratio = config.get('pruning', {}).get('prune_ratio', 0)
    revive_ratio = config.get('revival', {}).get('revive_ratio', 0)
    
    # Timestamp: MesDía_HoraMinuto (ej: 1219_1530)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    
    # Construir el nombre (puedes usar f-strings)
    # Ejemplo: ViT-B_smTrue_lr1e-4_p0.5_1219_1530
    run_name = f"{model_name}_sm{simulate_revival}_{distill_mode}dist_lr{lr}_p{prune_ratio}_r{revive_ratio}_{timestamp}"
    
    return run_name

def get_tokens_per_block(pruning_ratio, revive_ratio):
    pruning_ratio = np.array(pruning_ratio)
    revive_ratio = np.array(revive_ratio)

    N_TOKENS_INIT = 196
    n_tokens_alive = N_TOKENS_INIT
    n_tokens_dead = 0
    tokens_per_block = []
    for _ in range(12):
        # Pruning
        n_tokens_alive = int((pruning_ratio * n_tokens_alive).round())
        n_tokens_dead = N_TOKENS_INIT - n_tokens_alive

        # Revival
        n_tokens_alive += int((revive_ratio * n_tokens_dead).round())
        n_tokens_dead = N_TOKENS_INIT - n_tokens_alive

        tokens_per_block.append(int(n_tokens_alive))
    return tokens_per_block
    
# ----------------------------------------
# Main function
# ----------------------------------------
def main(config):
    # --- Seed
    L.seed_everything(config['seed'], workers=True, verbose=False)
    print(f"🌱 Seed fijada en {config['seed']}")

    # Map YAML sections to local variables
    model_cfg = config['model']
    run_cfg = config['run']
    pruning_cfg = config['pruning']
    revival_cfg = config['revival']
    dataset_cfg = config['dataset']
    mixup_cfg = config['mixup']

    # --- WandB logger
    experiment_name = generate_run_name(config)
    print(f"🚀 Iniciando experimento: {experiment_name}\n")
    wandb_logger = WandbLogger(
        project=run_cfg['wandb_project'],
        name=experiment_name,
        save_dir='wandb_logs'
    )

    # --- Dataset
    datamodule = load_dataset(
        dataset_cfg['name'],
        dataset_cfg['data_dir'],
        run_cfg['batch_size'],
        model_cfg['image_size'],
    )

    # --- Pruner & Revivor
    pruner = HeuristicPruner(criterion=pruning_cfg['pruning_criterion'])
    revivor = AffinityRevivor(criterion=revival_cfg['revival_criterion'])
    token_manager = TokenManager(
        pruner=pruner,
        revivor=revivor,
        prune_ratio_or_n_tokens=get_tokens_per_block(pruning_cfg['prune_ratio'], revival_cfg['revive_ratio']) if run_cfg['simulate_revival'] else pruning_cfg['prune_ratio'],
        revive_ratio_or_n_tokens=revival_cfg['revive_ratio'],
    )

    # --- Modelo
    model = load_model(
        model_name=model_cfg['name'],
        num_classes=datamodule.num_classes,
        pretrained=model_cfg['pretrained'],
        checkpoint_path=model_cfg['checkpoint_path'],
        token_manager=token_manager,
        can_tokens_revive=run_cfg['can_tokens_revive'],
    )

    # --- Teacher (opcional)
    teacher = create_model(
        "deit_small_patch16_224.fb_in1k",
        pretrained=True,
    )

    # --- Lightning Module
    lightning_module = TokenAdaptationModule(
        model=model,
        lr=run_cfg['lr'],
        mixup_config=mixup_cfg,
        teacher=teacher,
        distill_config=run_cfg['distill'],
    )

    # --- Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'checkpoints/{experiment_name}/',
        filename='{epoch}-{val_acc1:.5f}',
        monitor='val_acc1',
        save_last=True,
        save_top_k=3,
        mode='max',
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # --- Trainer
    trainer = L.Trainer(
        accelerator='auto',
        devices='auto',
        max_epochs=run_cfg['max_epochs'],
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=run_cfg['log_every_n_steps'],
        accumulate_grad_batches=run_cfg['accumulate_grad_batches'],
        val_check_interval=run_cfg['val_check_interval'],
    )

    # --- Train + Validate
    trainer.fit(lightning_module, datamodule=datamodule)

    # --- Evaluate
    print("\n✅ Evaluando modelo final...")
    trainer.validate(lightning_module, datamodule=datamodule)

# ----------------------------------------
# CLI
# ----------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TokenAdaptation from YAML config")
    parser.add_argument("--config", "-c", default="train_config.yaml", help="Ruta al archivo YAML de configuración")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    main(config)
