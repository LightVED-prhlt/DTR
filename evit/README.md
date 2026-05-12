# EViT (subfolder)

This folder contains the **EViT** (ICLR 2022) code used in this repository.

- Original repo (installation/conda environment): https://github.com/youweiliang/EViT

---

## 1) Dataset: ImageNet

The example scripts use `--data-path` pointing to the ImageNet root with this structure:

```
<IMAGENET_DIR>/
  train/<class>/*.JPEG
  val/<class>/*.JPEG
```

In this repo, adjust the `datapath` variable at the top of:
- [finetune.sh](finetune.sh)
- [evaluate.sh](evaluate.sh)

(Optional) If training in distributed mode, it is also present in `run_code.sh`.

---

## 2) Weights (downloads)

### 2.1 Base weights (required for training)

The [finetune.sh](finetune.sh) script loads a DeiT-Small base checkpoint via `--finetune`:

| Use | Where to save it | Link |
| --- | --- | --- |
| DeiT-Small (backbone) | `evit/checkpoints/deit_small_patch16_224-cd65a155.pth` | https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth |

Note: if `evit/checkpoints/` does not exist, create it.

### 2.2 Model Zoo (EViT pre-trained on ImageNet)

Short table (see full list in the original repo, **Model Zoo** section):

| Token fusion (`--fuse_token`) | Keep rate (`--base_keep_rate`) | w/ DTR | URL |
| --- | --- | --- | --- |
| ✓ | 0.5 | ✗ | [OneDrive](https://upvedues-my.sharepoint.com/:f:/g/personal/ljmarten_upv_edu_es/IgB3GPR5dzXiQpuSRHn9MEfpAVKkFk4ywTLhr161FKvCYiY?e=ekBsJQ) |
| ✓ | 0.6 | ✗ | [OneDrive](https://upvedues-my.sharepoint.com/:f:/g/personal/ljmarten_upv_edu_es/IgDW6TiIKiNrTLfhknei8VdBAR4MDS2nJ_5mtQdVNP-z5a4?e=aVoqJO) |
| ✓ | 0.7 | ✗ | [OneDrive](https://upvedues-my.sharepoint.com/:f:/g/personal/ljmarten_upv_edu_es/IgCD9E5YYpSuQofR94O_Gu12AVRUL_OEHyO_u85HsN8BWIw?e=mYslBM) |
| ✓ | 0.5 | ✓ | [OneDrive](https://upvedues-my.sharepoint.com/:f:/g/personal/ljmarten_upv_edu_es/IgDMstTMnBGIRIiUPajN5WbeASTSWsAs4pnpb93ON3Das8E?e=2rfXhf) |
| ✓ | 0.6 | ✓ | [OneDrive](https://upvedues-my.sharepoint.com/:f:/g/personal/ljmarten_upv_edu_es/IgDzG929IfJqToiFOqKJTaw1Ac2JjYHbQ_Om0SvmmMJ2wMw?e=xi8tp3) |
| ✓ | 0.7 | ✓ | [OneDrive](https://upvedues-my.sharepoint.com/:f:/g/personal/ljmarten_upv_edu_es/IgD5n2DIzd7ISamKjau2TeiDAU069UL_C16bs84M4pNeIs8?e=OWU7RZ) |

---

## 3) How to train (finetune)

The example script is [finetune.sh](finetune.sh).

1) Edit at the top of the file:
- `datapath="<IMAGENET_DIR>/"`
- `ckpt="./checkpoints/deit_small_patch16_224-cd65a155.pth"` (downloaded in section 2.1)

2) Typical settings:
- `--base_keep_rate 0.7` (the lower, the more aggressive the token dropping/fusion)
- `--fuse_token` (removing this flag disables token fusion)

3) Run from this folder (`mycode/evit`):

```bash
bash finetune.sh
```

Expected output:
- `--output_dir` points to `./finetune_log/...`
- The best checkpoint is saved as `checkpoint_best.pth` inside that directory

---

## 4) How to evaluate / inference

The example script is [evaluate.sh](evaluate.sh). It runs `main.py` in evaluation mode (`--eval`), loading a checkpoint via `--resume`.

Steps:
1) Adjust `datapath`.
2) Adjust which checkpoint to evaluate (e.g. by changing `p07_rkt` or using a different path):
   - checkpoints from your finetunes: `finetune_log/<exp>/checkpoint_best.pth`
   - or a checkpoint from the **Model Zoo** (downloaded from section 2.2)
3) Run:

```bash
bash evaluate.sh
```