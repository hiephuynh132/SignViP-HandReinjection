INFERENCE / EVALUATION RELEASE: WEIGHTS AND EXTERNAL INPUTS
Do not upload weights or RWTH video data to Git.
Fill DRIVE_URL below after uploading each permitted artifact.
SOURCE_REFERENCE identifies an artifact in the original research workspace;
it is not an existing download location in this source-only repository.
Users restore downloaded files to RESTORE_TO. Private workstation paths are
kept outside this repository in the local upload inventory.
The proposed best and standalone VAE have now been moved into RESTORE_TO locally.
SOURCE_REFERENCE entries below are historical references, not live locations.

1. PROPOSED MODEL (A12 / C5.15), selected step 282000
SOURCE_REFERENCE=./signvip/workspace_c5_15/C5.15_final_stage1_A0_best_vae/20260920-1230/best
RESTORE_TO=weights/proposed/best/
DRIVE_URL=TODO
Upload the entire best directory with its subfolders and metrics.json.
Keep appearance_encoder/, condition_encoder/, hamer_reinjector/, vae/ together.
This artifact is for inference/evaluation, not a full optimizer-state resume.

2. OPTIMIZED VAE (C5.13), selected step 195000
SOURCE_REFERENCE=./signvip/workspace_resume/C5.13_vae_reconstruction_hand4x_face4x_clip_dino_lpips/20260903-1245/best
RESTORE_TO=weights/optimized_vae/
DRIVE_URL=TODO
Preserve vae/config.json, vae/diffusion_pytorch_model.safetensors and metrics.json.
The proposed best package contains a vae/ folder; standalone upload is optional
for proposed-model inference. Use this standalone artifact to evaluate the
separately selected C5.13 VAE; do not assume both VAE folders are byte-identical.

3. TRAINING INITIALIZATION: NOT DISTRIBUTED
The historical A9 / C5.14 initialization checkpoint is excluded from this release.
It is not required by eval.sh. Training code/configs are retained for reference,
but reproducing the warm-started training requires this additional artifact.

4. PRETRAINED STABLE DIFFUSION v1.5
SOURCE_REFERENCE=./animatediff/models/StableDiffusion/stable-diffusion-v1-5
RESTORE_TO=weights/stable-diffusion-v1-5/
UPSTREAM_REFERENCE=TODO_OFFICIAL_PROVIDER_AND_REVISION
Preserve unet/, vae/, tokenizer/, text_encoder/, scheduler/ and their configs.
Prefer referencing the original model provider and its access/license terms.

5. PERCEPTUAL MODELS
MODEL_ID=openai/clip-vit-large-patch14
CACHE_REFERENCE=data/hf_cache/hub/models--openai--clip-vit-large-patch14
UPSTREAM_REFERENCE=TODO_OFFICIAL_PROVIDER_AND_REVISION
MODEL_ID=facebook/dinov2-small
CACHE_REFERENCE=data/hf_cache/hub/models--facebook--dinov2-small
UPSTREAM_REFERENCE=TODO_OFFICIAL_PROVIDER_AND_REVISION
LPIPS AlexNet weights are also needed; install lpips and populate its pretrained cache.

6. DATA / CONDITIONS (REQUIRED EXTERNAL INPUTS; NOT GIT ASSETS)
RGB_POSE_HDF5=data/rwth
HAMER_HDF5=data/hamer
MASK_CACHE=data/detail_masks
DWPOSE_CACHE=data/dwpose_keypoints
Both caches have been moved into these paths. RGB/pose and HaMeR HDF5 relocation
is pending filesystem-owner permission; do not archive those inputs until moved.
RAW_RWTH_VIDEOS=data/raw_rwth/videos
RAW_RWTH_ANNOTATIONS=data/raw_rwth/manual
Use the dataset provider's access terms for data redistribution.
Grounded-SAM2, HaMeR/MANO, DWPose detector weights belong to their own providers.
Their custom preprocessing scripts are in preprocessing/; those upstream packages
and model files are intentionally not embedded in this source snapshot.

7. FULL TRAINING RESUME: NOT DISTRIBUTED
Optimizer, scheduler, scaler and random-state checkpoints are excluded.
The released best/ artifact supports inference/evaluation, not full training resume.
