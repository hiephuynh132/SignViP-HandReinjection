# syntax=docker/dockerfile:1
ARG BASE_IMAGE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    MPLCONFIGDIR=/tmp/matplotlib \
    SD15_ROOT=/weights/stable-diffusion-v1-5 \
    PAPER_VAE_ROOT=/weights/optimized_vae \
    PAPER_INIT_ROOT=/weights/stage1_init \
    CHECKPOINT_DIR=/weights/proposed/best \
    DATA_ROOT=/data/rwth \
    HAMER_ROOT=/data/hamer \
    MASK_ROOT=/data/detail_masks \
    POSE_ROOT=/data/dwpose_keypoints \
    SIGNVIP_WORKSPACE_ROOT=/outputs/stage1 \
    OUTPUT_ROOT=/outputs \
    GPU_ID=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt constraints-docker.txt ./
# Constrain torch/torchvision so installing LPIPS cannot upgrade the CUDA stack.
RUN python -m pip install --no-cache-dir \
    -c constraints-docker.txt -r requirements.txt torchvision==0.22.1

COPY . .
RUN python tools/check_release.py \
    && python -c "import sys; sys.path.insert(0, 'signvip'); import train_stage1_multi_cond; import eval_C5_1_full_test"

CMD ["bash"]
