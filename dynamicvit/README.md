# DynamicViT (subfolder)

This folder contains the **DynamicViT** (NeurIPS 2021) code used in this repository.

- Original repo (installation/conda environment): https://github.com/raoyongming/DynamicViT

---

## 1) Dataset: ImageNet

The example scripts use `--data_path` pointing to the ImageNet root with this structure:

```
<IMAGENET_DIR>/
  train/<class>/*.JPEG
  val/<class>/*.JPEG
```

In this repo, adjust the `datapath` variable at the top of:
- [finetune.sh](finetune.sh)
- [evaluate.sh](evaluate.sh)

---

## 2) Weights (downloads)

### 2.1 Base weights (required for training)

`main.py` loads base weights from `./pretrained/` using fixed filenames.
For the default example (`--model deit-s`) you need:

| Use | Where to save it | Link |
| --- | --- | --- |
| DeiT-Small (backbone) | `dynamicvit/pretrained/deit_small_patch16_224-cd65a155.pth` | https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth |

### 2.2 Model Zoo (DynamicViT pre-trained on ImageNet)

Short table (see full list in the original repo, **Model Zoo** section):

| Name | rho | w/ DTR | URL |
| --- | --- | --- | --- | 
| DynamicViT-DeiT-S/0.5 | 0.5 | ✗ | [OneDrive](https://upvedues-my.sharepoint.com/:f:/g/personal/ljmarten_upv_edu_es/IgDxwZ6ChbUwS6AwBRkEhOK0ARvYaGc5z_ZeBILybyoZRzY?e=TtAI88) | 
| DynamicViT-DeiT-S/0.6 | 0.6 | ✗ | [OneDrive](https://upvedues-my.sharepoint.com/:f:/g/personal/ljmarten_upv_edu_es/IgBO5_BFLFweTq9j1aJBPNJhAQi9FFNkBp7bmSvaQcEy2cs?e=0uyp6u) | 
| DynamicViT-DeiT-S/0.7 | 0.7 | ✗ | [Google Drive](https://drive.google.com/file/d/1H5kHHagdqo4emk9CgjfA7DA62XJr8Yc1/view) |
| DynamicViT-DeiT-S/0.5 | 0.5 | ✓ | [OneDrive](https://upvedues-my.sharepoint.com/:f:/g/personal/ljmarten_upv_edu_es/IgDkHinf8oC3TJ9m3FP6pu9fAVQ67TYaI7G8QiIWhlm9Z3c?e=vBk5xs) | 
| DynamicViT-DeiT-S/0.6 | 0.6 | ✓ | [OneDrive](https://upvedues-my.sharepoint.com/:f:/g/personal/ljmarten_upv_edu_es/IgB-rqvLPZemQ4TXvE-j2ShgAXsf-AQSp6raoVLQT5Yeaaw?e=xmFzuG) | 
| DynamicViT-DeiT-S/0.7 | 0.7 | ✓ | [OneDrive](https://upvedues-my.sharepoint.com/:f:/g/personal/ljmarten_upv_edu_es/IgCAevoehYeRS6QqN4Ro407YAS_nPAs6_ObMXjVkJoAjW50?e=vDXIrK) |

---

## 3) How to train (finetune)

The example script is [finetune.sh](finetune.sh).

1) Edit at the top of the file:
- `datapath="<IMAGENET_DIR>/"`
- `base_rate=0.7` (keeping ratio; the lower, the more aggressive the pruning)

2) Run from this folder (`mycode/dynamicvit`):

```bash
bash finetune.sh
```

Expected output:
- `--output_dir` points to something like `logs/...`
- Checkpoints are typically saved inside the output directory (e.g. `checkpoint-best.pth`)

Note: `finetune.sh` has a commented-out "WITHOUT REVIVING" block and an active "WITH REVIVING" block (flag `--with_dtr`).

---

## 4) How to evaluate / inference

The example script is [evaluate.sh](evaluate.sh). By default it runs:

```bash
python infer.py --data_path <IMAGENET_DIR>/ --model deit-s --model_path <CKPT.pth> --base_rate 0.7
```

Steps:
1) Adjust `datapath`.
2) Adjust `model_path` to point to the checkpoint you want to evaluate (e.g. a `checkpoint-best.pth` produced during training).
3) If your checkpoint corresponds to the DTR variant, keep `--with_dtr`; otherwise, remove it.
4) Run:

```bash
bash evaluate.sh
```