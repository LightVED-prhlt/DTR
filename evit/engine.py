# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for adjusting keep rate and visualization -- Youwei Liang
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional
from pathlib import Path

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils

from helpers import adjust_keep_rate
from visualize_mask import get_real_idx, mask, save_img_batch, mask_with_alpha
import os
import glob
from torchvision import transforms as T
from torchvision.utils import save_image
from PIL import Image as PILImage
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
#                     data_loader: Iterable, optimizer: torch.optim.Optimizer,
#                     device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
#                     model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
#                     writer=None,
#                     set_training_mode=True,
#                     args=None):
#     model.train(set_training_mode)
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     header = 'Epoch: [{}]'.format(epoch)
#     print_freq = 200
#     log_interval = 100
#     it = epoch * len(data_loader)
#     ITERS_PER_EPOCH = len(data_loader)
#     UPDATE_FREQ = args.update_freq

#     base_rate = args.base_keep_rate

#     for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
#         samples = samples.to(device, non_blocking=True)
#         targets = targets.to(device, non_blocking=True)

#         keep_rate = adjust_keep_rate(it, epoch, warmup_epochs=args.shrink_start_epoch,
#                                          total_epochs=args.shrink_start_epoch + args.shrink_epochs,
#                                          ITERS_PER_EPOCH=ITERS_PER_EPOCH, base_keep_rate=base_rate)

#         if mixup_fn is not None:
#             samples, targets = mixup_fn(samples, targets)

#         with torch.cuda.amp.autocast():
#             outputs = model(samples, keep_rate)
#             loss = criterion(samples, outputs, targets)

#         loss_value = loss.item()

#         if not math.isfinite(loss_value):
#             print("Loss is {}, stopping training".format(loss_value))
#             sys.exit(1)

#         optimizer.zero_grad()

#         # this attribute is added by timm on one optimizer (adahessian)
#         is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
#         loss_scaler(loss, optimizer, clip_grad=max_norm,
#                     parameters=model.parameters(), create_graph=is_second_order)

#         torch.cuda.synchronize()
#         if model_ema is not None:
#             model_ema.update(model)

#         metric_logger.update(loss=loss_value)
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])

#         # if torch.distributed.get_rank() == 0 and it % log_interval == 0:
#         #     writer.add_scalar('loss', loss_value, it)
#         #     writer.add_scalar('lr', optimizer.param_groups[0]["lr"], it)
#         #     writer.add_scalar('keep_rate', keep_rate, it)
#         it += 1

#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, keep_rate

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[torch.nn.Module] = None, mixup_fn: Optional[object] = None,
                    writer=None,
                    set_training_mode=True,
                    args=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    print_freq = 200
    log_interval = 100
    ITERS_PER_EPOCH = len(data_loader)
    UPDATE_FREQ = args.update_freq
    base_rate = args.base_keep_rate

    # Limpiamos gradientes al inicio del epoch
    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # it: contador global de iteraciones para schedulers y logging
        it = epoch * ITERS_PER_EPOCH + data_iter_step
        
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Ajuste de keep_rate (específico para arquitecturas con pruning/shrinkage)
        keep_rate = adjust_keep_rate(it, epoch, warmup_epochs=args.shrink_start_epoch,
                                     total_epochs=args.shrink_start_epoch + args.shrink_epochs,
                                     ITERS_PER_EPOCH=ITERS_PER_EPOCH, base_keep_rate=base_rate)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples, keep_rate)
            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # 1. Normalizar la pérdida según la frecuencia de acumulación
        # Esto asegura que el gradiente tenga la magnitud correcta
        loss /= UPDATE_FREQ

        # 2. Determinar si toca actualizar pesos en este paso
        is_update_step = (data_iter_step + 1) % UPDATE_FREQ == 0
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        # 3. Llamada al scaler. 
        # Si is_update_step es False, solo hará el backward (acumula).
        # Si is_update_step es True, hará backward, unscale, clip, step y update.
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order,
                    update_grad=is_update_step)

        # 4. Tareas post-actualización
        if is_update_step:
            optimizer.zero_grad()
            if model_ema is not None:
                model_ema.update(model)

        # Loggear métricas
        # Usamos el valor real de la pérdida (multiplicado de nuevo) para el log si prefieres ver la escala original
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # if torch.distributed.get_rank() == 0 and it % log_interval == 0:
        #     if writer is not None:
        #         writer.add_scalar('loss', loss_value, it)
        #         writer.add_scalar('lr', optimizer.param_groups[0]["lr"], it)
        #         writer.add_scalar('keep_rate', keep_rate, it)

    # Reunir estadísticas de todos los procesos (distribuido)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, keep_rate

@torch.no_grad()
def evaluate(data_loader, model, device, keep_rate=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images, keep_rate)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def get_acc(data_loader, model, device, keep_rate=None, tokens=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images, keep_rate, tokens)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return metric_logger.acc1.global_avg

@torch.no_grad()
def distributed_visualize_mask(data_loader, model, device, output_dir, n_visualization, fuse_token, keep_rate=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Visualize:'
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    mean = torch.tensor(IMAGENET_DEFAULT_MEAN, device=device).reshape(3, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD, device=device).reshape(3, 1, 1)

    # switch to evaluation mode
    model.eval()

    ii = 0
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        B = images.size(0)

        with torch.cuda.amp.autocast():
            output, idx = model(images, keep_rate, get_idx=True)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # denormalize
        images = images * std + mean

        idxs = get_real_idx(idx, fuse_token)
        for jj, idx in enumerate(idxs):
            masked_img = mask(images, patch_size=16, idx=idx)
            save_img_batch(masked_img, output_dir, file_name='img_{}' + f'_l{jj}.jpg', start_idx=world_size * B * ii + rank * B)

        save_img_batch(images, output_dir, file_name='img_{}_a.jpg', start_idx=world_size * B * ii + rank * B)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.synchronize_between_processes()
        ii += 1
        if world_size * B * ii >= n_visualization:
            break

    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def visualize_mask(data_loader, model, device, output_dir, n_visualization, fuse_token, keep_rate=None, alpha=None):
    # Create a new data loader from a path to an image folder
    dataset = ImageFolder(
        root="imgs/",
        transform=T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ]),
    )
    data_loader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=1
    )
    
    output_dir = "output_images/"
    n_visualization = 10
    device = "cpu"
    
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Visualize:'
    # Ejecutar siempre en un único proceso/GPU
    rank = 0
    world_size = 1
    mean = torch.tensor(IMAGENET_DEFAULT_MEAN, device=device).reshape(3, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD, device=device).reshape(3, 1, 1)

    # switch to evaluation mode
    model.eval()
    model.to(device)

    ii = 0
    total_processed = 0
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        B = images.size(0)

        with torch.cuda.amp.autocast():
            output, idx = model(images, keep_rate, get_idx=True)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # denormalize
        # images = images * std + mean

        # idxs = get_real_idx(idx, fuse_token)
        idxs = idx
        
        # guardar máscaras por imagen
        for jj, idx in enumerate(idxs):
            masked_img = mask_with_alpha(images, idx=idx, patch_size=16, alpha=0.2)
            start_idx = total_processed
            save_img_batch(masked_img, output_dir, file_name='img_{}' + f'_l{jj}.jpg', start_idx=start_idx)

        # guardar imagen original denormalizada
        save_img_batch(images, output_dir, file_name='img_{}_a.jpg', start_idx=total_processed)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        ii += 1
        total_processed += B
        if total_processed >= n_visualization:
            break

    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def visualize_mask_from_folder(images_dir, model, device, output_dir, n_visualization, fuse_token, keep_rate=None,
                               input_size=224, batch_size=16, patch_size=16):
    """
    Visualiza máscaras a partir de imágenes en una carpeta y guarda SOLO las imágenes con la máscara aplicada.

    Args:
        images_dir (str): ruta a la carpeta con imágenes (jpg/png).
        model: modelo que devuelve (output, idx) cuando se llama con get_idx=True.
        device: dispositivo ('cuda'/'cpu').
        output_dir (str): carpeta donde guardar las imágenes enmascaradas.
        n_visualization (int): número máximo de imágenes a procesar.
        fuse_token (bool): parámetro pasado a get_real_idx.
        keep_rate (float): keep rate a pasar al modelo (puede ser None).
        input_size (int): tamaño de entrada para redimensionar.
        batch_size (int): tamaño de batch para procesamiento.
        patch_size (int): tamaño de parche que usa la función `mask`.

    Returns:
        List[str]: rutas de los archivos guardados.
    """
    # listar imágenes
    exts = ['*.jpg']
    files = []
    for e in exts:
        files.extend(sorted(glob.glob(os.path.join(images_dir, e))))
    files = files[:n_visualization]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # transform para obtener tensor en [0,1]
    if input_size > 32:
        size = int((256 / 224) * input_size)
        transform_raw = T.Compose([T.Resize(size), T.CenterCrop(input_size), T.ToTensor()])
    else:
        transform_raw = T.Compose([T.Resize(input_size), T.ToTensor()])

    mean = torch.tensor(IMAGENET_DEFAULT_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD).view(3, 1, 1)

    saved_files = []
    for i in range(0, len(files), batch_size):
        batch_paths = files[i:i + batch_size]
        imgs = []
        names = []
        for p in batch_paths:
            try:
                pil = PILImage.open(p).convert('RGB')
            except Exception:
                continue
            t = transform_raw(pil)
            imgs.append(t)
            names.append(Path(p).stem)

        if len(imgs) == 0:
            continue

        imgs_raw = torch.stack(imgs, dim=0)  # [B,3,H,W], range [0,1]
        # normalizar para el modelo
        imgs_norm = (imgs_raw - mean) / std
        imgs_norm = imgs_norm.to(device)

        with torch.cuda.amp.autocast():
            out = model(imgs_norm, keep_rate, get_idx=True)
            # algunos modelos devuelven únicamente (out, idx)
            if isinstance(out, tuple) and len(out) >= 2:
                _, idx = out[0], out[1]
            else:
                # fallback: intentar desempacar
                try:
                    _, idx = out
                except Exception:
                    raise RuntimeError('El modelo no devolvió índices (idx) con get_idx=True')

        idxs = get_real_idx(idx, fuse_token)

        # mover imgs_raw a device para operar con mask, luego guardar en CPU
        imgs_raw_device = imgs_raw.to(device)
        for bi, nm in enumerate(names):
            idx_single = idxs[bi].unsqueeze(0)
            masked = mask(imgs_raw_device[bi:bi + 1], idx_single, patch_size=patch_size)  # [1,3,H,W]
            out_name = f"{nm}_masked.jpg"
            out_path = os.path.join(output_dir, out_name)
            # guardar imagen (pasar a CPU)
            save_image(masked.squeeze(0).cpu(), out_path)
            saved_files.append(out_path)

    return saved_files
