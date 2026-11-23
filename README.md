# SignViP-HandReinjection

Docker build, GPU selection and mount instructions: **[DOCKER.md](DOCKER.md)**.
Public source/config/metrics use portable paths. Workstation upload references
are kept outside this repository; run `python tools/check_release.py` before publishing.

Source preparation for **Improving Fine-Grained Sign Language Video Generation
with Multi-Level 3D Hand Feature Reinjection**.

Paper scope: RWTH-PHOENIX-Weather 2014T, 128×128, frame-level Stage I generation.
This package includes the optimized-VAE training code, three-site HaMeR residual
scalar reinjection, region-weighted denoising, CLIP/DINO supervision, full-test
evaluation, paper figures and recorded metrics. Thesis VSL-UTE-1.0 data/configs,
training workspaces, GPU watchers and Telegram credentials are not included.

## Files

| Location | Purpose |
|---|---|
| `signvip/models/` | Shared model implementations, including `hamer_reinjection.py` |
| `signvip/train_stage1_multi_cond.py` | Stage I training and checkpoint resume |
| `signvip/train_vae_decoder_reconstruction.py` | Hand/face-aware VAE decoder fine-tuning |
| `signvip/perceptual_losses.py` | CLIP/DINO and regional reconstruction utilities |
| `signvip/signdatasets/hdf5_sign_lang_video.py` | Paired RGB/pose/HaMeR HDF5 input |
| `signvip/eval_C5_1_full_test.py` | PSNR/SSIM/LPIPS and regional full-test metrics |
| `signvip/configs/paper/` | Flattened configs with portable input paths |
| `scripts/` | Train VAE, train proposal and evaluate commands |
| `autoencoder_used/` | VAE reconstruction evaluation/visualization utilities |
| `preprocessing/` | Original project-specific dataset/condition adapters |
| `provenance/` | Original config inheritance chain, saved run config and copy hashes |
| `results/` | Selected-checkpoint metadata and full-test metrics |
| `assets/` | Five original paper figure assets |
| `WEIGHT_PATHS.md` | Relative source references, restore layout and Drive URL fields |

## Prepare weights and inputs

This release targets inference/evaluation of the paper's selected model, not
end-to-end training reproduction. Upload the proposed C5.15 `best/` folder to
Drive; optionally provide the standalone C5.13 VAE for VAE comparisons. Replace
their TODO fields in `WEIGHT_PATHS.md` with download links. Public pretrained
models should be obtained from their original providers, not re-uploaded to Drive.
Preserve each folder's internal layout. The historical C5.14 initialization and
full optimizer-state checkpoints are not distributed.

Expected local layout after download:

```text
weights/
  stable-diffusion-v1-5/{unet,vae,tokenizer,text_encoder,scheduler}/
  optimized_vae/vae/
  proposed/best/{appearance_encoder,condition_encoder,hamer_reinjector,vae}/
data/
  rwth/{train,dev,test}/shard_0001.hdf5
  hamer/{train,dev,test}/shard_0001.hdf5
  detail_masks/{train,dev,test}/*.npz
  dwpose_keypoints/test/*.npz
```

Set absolute paths if using a different layout:

```bash
export SD15_ROOT="$PWD/weights/stable-diffusion-v1-5"
export PAPER_VAE_ROOT="$PWD/weights/optimized_vae"
export PAPER_INIT_ROOT="$PWD/weights/stage1_init"
export DATA_ROOT="$PWD/data/rwth"
export HAMER_ROOT="$PWD/data/hamer"
export MASK_ROOT="$PWD/data/detail_masks"
export POSE_ROOT="$PWD/data/dwpose_keypoints"
```

## Environment and commands

Use Python 3.10 or newer. Install a matching CUDA build of PyTorch/torchvision for
your GPU, then install `requirements.txt`. `requirements.upstream.txt` is retained
for historical reference; it is not a verified fresh-install lockfile. In
particular, do not blindly install its historical torch/xformers pins on newer
GPUs. The source snapshot has not been certified in a fresh environment.

```bash
python -m pip install -r requirements.txt

# Optional: reproduce optimized VAE training (paper selection: step 195000).
GPU_ID=0 bash scripts/train_vae.sh

# Reference only: requires the C5.14 initialization, NOT distributed here.
GPU_ID=0 bash scripts/train_proposal.sh

# Reproduce the supplied proposed model's test evaluation.
GPU_ID=0 bash scripts/eval.sh

# Compare pretrained and optimized VAE reconstruction on all test frames.
GPU_ID=0 bash scripts/eval_vae.sh

# Small inference check first, using a separate output folder.
GPU_ID=0 OUTPUT_DIR="$PWD/outputs/eval_smoke" \
  bash scripts/eval.sh --max-samples 1 --save-previews 1
```

For VAE reconstruction evaluation, use `autoencoder_used/eval_autoencoders_used.py
--help`. Only the SD1.5 Diffusers backend is in scope; historical optional
autoencoder backends require packages not bundled here.

## Reported model and protocol

The proposed paper model is **A12 / C5.15**, selected checkpoint step **282000**.
The optimized VAE is **C5.13**, selected at step **195000**. `best/metrics.json`
records the selected A12 checkpoint's rule as `global_ssim`; the saved run config
later lists balanced hand/nonhand SSIM. This historical mismatch is retained in
`provenance/` instead of silently claiming a single selection history. The
portable training config uses the selected artifact's recorded global-SSIM rule.

Full-test evaluation uses all 642 test videos / 64,627 frames, reference frame 0,
DDIM 50 steps, CFG 3.5, inference chunks of 5 and seed 42. The selected model has:

| Metric | Recorded result |
|---|---:|
| Global PSNR | 24.545763 |
| Global SSIM | 0.863482 |
| Global LPIPS | 0.06484 (rounded in paper) |
| Hand SSIM | 0.708428 |
| Hand LPIPS | 0.134065 |

The training config preserves paper SNR weighting (`snr_gamma=5`), separate
adapter/encoder base learning rates (`1e-4` / `5e-7`), cosine scheduler, effective
batch 8, frozen VAE/U-Net and trainable appearance/condition encoders plus
reinjection. It is warm-started from the recorded parent run, not trained from
scratch. The parent run's earlier training is not reproduced by one command.
The parent checkpoint is not distributed in this inference/evaluation release;
`scripts/train_proposal.sh` is not runnable with the released best weights alone.

Portable VAE training uses a 200k-step horizon so the reported best at 195k is
reachable; the archived original experiment config started with a 50k horizon
and was extended through subsequent runs. Periodic checkpoint retention in the
portable Stage I config keeps one last state; best inference modules are separate.

## Reproducibility status

The Python files are copied from the current working source, including shared
optional research branches and later checkpoint reliability fixes. They are
not claimed to be a pristine historical commit from the date of paper training.
Portable configs explicitly control the paper path; arbitrary optional branches
are outside the supported release scope. Both original-source and sanitized
release hashes are recorded for copied files. Archived configs and metrics have
workstation paths replaced by relative references; numeric results are preserved.
No training, GPU inference or weight upload is performed by
the packaging step.

Preparation checks passed: copied-source hashes, Python syntax, YAML parsing,
portable OmegaConf resolution, shell syntax, absence of bundled weights/data,
and importing the packaged trainer/evaluator with the original workspace's
dependency overlays. Import checks do not validate GPU inference or numerical
reproduction in a newly installed environment.

The HDF5 loader is the supported dataset entry point. Other lazy dataset exports
refer to optional original SignViP modules not needed for this paper path.
Custom raw-data scripts need the upstream repositories described in
`preprocessing/README.md`; they are not standalone replacements for those repos.

Before publication, fill the Drive URLs, author/citation details and review
`THIRD_PARTY_NOTICES.md`. This directory is prepared locally; no Git repository,
remote, commit or public upload is created automatically.
