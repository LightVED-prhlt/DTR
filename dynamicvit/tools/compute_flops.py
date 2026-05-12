#!/usr/bin/env python3
"""Calculator de FLOPs/MACs para VisionTransformer con pruning dinámico.

Uso:
  python tools/compute_flops.py            # ejecuta ejemplo por defecto
  python tools/compute_flops.py --img-size 224 --patch-size 16

El script importa `VisionTransformerDiffPruning` desde `models.dyvit` si está
presente y extrae `patch_embed.num_patches`, `embed_dim`, `blocks`,
`pruning_loc` y `token_ratio` para calcular FLOPs por bloque.
"""
import math
import argparse
import importlib
import sys

try:
    import torch
except Exception:
    torch = None


def conv_patch_embed_macs_from_module(model, img_size=None, patch_size=None):
    pe = model.patch_embed
    # obtener tamaño y kernel
    if hasattr(pe, 'img_size') and pe.img_size is not None:
        H = pe.img_size[0]
        W = pe.img_size[1]
    else:
        H = W = img_size or 224

    k = patch_size or (pe.patch_size[0] if isinstance(pe.patch_size, (tuple, list)) else pe.patch_size)
    Cin = pe.proj.weight.shape[1]
    Cout = pe.proj.weight.shape[0]
    ph = H // k
    pw = W // k
    N = ph * pw
    macs = N * Cout * Cin * (k * k)
    return macs, N


def attn_macs_per_block(N, C):
    # MACs counting multiply-add as 1 MAC. Convert luego a FLOPs multiplicando por 2.
    # QKV + out proj ~= 4*N*C^2; attention (QK and attn*V) ~= 2*N^2*C
    macs = 4 * N * C * C + 2 * (N * N) * C
    return macs


def mlp_macs_per_block(N, C, mlp_ratio=4.0):
    # two linear layers: N*C*(r*C) + N*(r*C)*C = 2 * r * N * C^2
    r = mlp_ratio
    macs = 2 * r * N * C * C
    return macs


def block_macs(N, C, mlp_ratio=4.0):
    return attn_macs_per_block(N, C) + mlp_macs_per_block(N, C, mlp_ratio)


def predictor_macs_per_stage(N, C):
        """Estima MACs del PredictorLG aplicado a N tokens con dimensión C.

        Basado en la implementación de `PredictorLG` en models/dyvit.py:
        - in_conv: LayerNorm + Linear(C->C)  -> aproximamos al coste de Linear: C*C
        - out_conv: Linear(C->C/2) + Linear(C/2->C/4) + Linear(C/4->2)
            -> MACs ~= 0.5*C^2 + 0.125*C^2 + 0.5*C
        - agregación global sobre tokens (multiplicación por policy y suma) ~ N*C

        Devuelve MACs (no FLOPs).
        """
        # por token
        macs_per_token = 1.0 * C * C  # in_conv linear C->C
        macs_per_token += 0.5 * C * C  # out_conv first linear C->C/2
        macs_per_token += 0.125 * C * C  # out_conv second linear C/2->C/4
        macs_per_token += 0.5 * C  # out_conv last linear C/4->2 (≈0.5*C)
        # global aggregation cost (approx)
        agg_per_token = 1.0 * C
        total = N * (macs_per_token + agg_per_token)
        return total


def total_macs_for_model(model, include_cls=True, training=False):
    # conv patch embed
    conv_macs, N0 = conv_patch_embed_macs_from_module(model)
    macs_total = conv_macs

    # inferir parámetros desde el modelo
    C = model.embed_dim
    depth = len(model.blocks)

    # inferir mlp_ratio desde el primer bloque si es posible
    try:
        fc1_shape = model.blocks[0].mlp.fc1.weight.shape
        mlp_hidden = fc1_shape[0]
        mlp_in = fc1_shape[1]
        mlp_ratio = mlp_hidden / mlp_in
    except Exception:
        mlp_ratio = 4.0

    pruning_loc = getattr(model, 'pruning_loc', []) or []
    token_ratio = getattr(model, 'token_ratio', []) or []

    N = N0
    macs_per_block = []
    p_count = 0
    for i in range(depth):
        if i in pruning_loc:
            # PredictorLG se ejecuta sobre los tokens actuales antes del pruning.
            pred_macs = predictor_macs_per_stage(N, C)
            # sumar coste del predictor
            macs_total += pred_macs

            # inference deterministic pruning: reducir tokens antes de calcular el bloque
            r = float(token_ratio[p_count])
            N_keep = math.ceil(N * r)
            N_block = N_keep + (1 if include_cls else 0)
            macs = block_macs(N_block, C, mlp_ratio)
            macs_per_block.append(macs)
            macs_total += macs
            N = N_keep
            p_count += 1
        else:
            N_block = N + (1 if include_cls else 0)
            macs = block_macs(N_block, C, mlp_ratio)
            macs_per_block.append(macs)
            macs_total += macs

    return {
        'macs_total': macs_total,
        'flops_total': 2 * macs_total,
        'macs_per_block': macs_per_block,
        'num_patches_initial': N0,
        'mlp_ratio': mlp_ratio,
        'depth': depth,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models.dyvit', help='module path where model is located')
    parser.add_argument('--class', dest='cls', default='VisionTransformerDiffPruning', help='class name')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--patch-size', type=int, default=16)
    parser.add_argument('--embed-dim', type=int, default=384)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--pruning-loc', nargs='*', type=int, default=[3,6,9])
    parser.add_argument('--token-ratio', nargs='*', type=float, default=[0.7, 0.7, 0.7])
    parser.add_argument('--no-cls', action='store_true', help='exclude CLS token in attention FLOPs')
    args = parser.parse_args()

    # intentar importar la clase del repo
    try:
        print('Intentando importar', args.cls, 'desde', args.model)
        mod = importlib.import_module(args.model)
        print(mod)
        ModelClass = getattr(mod, args.cls)
        print(ModelClass)
    except Exception:
        ModelClass = None

    if ModelClass is not None:
        # instanciar modelo con pruning args
        try:
            model = ModelClass(img_size=args.img_size, patch_size=args.patch_size,
                               embed_dim=args.embed_dim, depth=args.depth,
                               pruning_loc=args.pruning_loc, token_ratio=args.token_ratio)
        except Exception:
            # fallback: instanciar sin pruning y luego setear atributos
            model = ModelClass(img_size=args.img_size, patch_size=args.patch_size,
                               embed_dim=args.embed_dim, depth=args.depth,
                               pruning_loc=args.pruning_loc, token_ratio=args.token_ratio)

        res = total_macs_for_model(model, include_cls=(not args.no_cls))
        print('Modelo:', args.cls)
        print('Img size:', args.img_size, 'Patch size:', args.patch_size)
        print('Emb dim:', model.embed_dim, 'Depth:', res['depth'])
        print('Num patches (init):', res['num_patches_initial'])
        print('MLP ratio (inferred):', res['mlp_ratio'])
        print('MACs total: %.6f G' % (res['macs_total'] / 1e9))
        print('FLOPs total: %.6f G' % (res['flops_total'] / 1e9))
        print('MACs per block:', [m / 1e6 for m in res['macs_per_block']])
    else:
        print('No se pudo importar la clase', args.cls, 'desde', args.model)
        sys.exit(1)


if __name__ == '__main__':
    main()
