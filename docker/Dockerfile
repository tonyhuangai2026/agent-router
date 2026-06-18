# Local containerized training image (no SageMaker) — Tech Design §2/§4.
#
# Base ships a CUDA-matched PyTorch build (same 2.6/cu126 stack as the
# SageMaker DLC). With `--gpus` it runs on GPU; without, the same torch
# transparently falls back to CPU. CRITICAL: we never pip-install or reinstall
# torch — src/requirements.txt intentionally omits torch so the base image's
# GPU build is not clobbered to a CPU-only wheel (Tech Design §1, §7).
#
# NOTE: the floating tag `2.6-cuda12.6-cudnn9-runtime` is not published on
# Docker Hub; we use the fully-qualified patch tag `2.6.0-cuda12.6-cudnn9-runtime`
# — the same PyTorch 2.6 / CUDA 12.6 / cuDNN9 runtime stack. This is the
# "邻近 tag (如 2.6.0-cuda12.6)" fallback prescribed by Tech Design §7.
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

WORKDIR /app

# Project code only — no outputs/ and no raw data are baked into the image
# (those are mounted at runtime per Tech Design §3). Keep the in-repo layout
# (src/, data/) so module imports resolve exactly as outside the container.
COPY src/ /app/src/
COPY data/prepare_data.py /app/data/prepare_data.py
COPY data/prepare_data_streaming.py /app/data/prepare_data_streaming.py

# LoRA-SFT stack layered on top of the base torch + numpy/matplotlib for the
# evaluate report plots. torch is deliberately absent from requirements.txt.
RUN pip install --no-cache-dir -r src/requirements.txt \
    && pip install --no-cache-dir numpy matplotlib

# Placeholder mount points for prepared data / model / report (Tech Design §3).
RUN mkdir -p /work/prepared /work/model /work/report

# Stage dispatcher: prepare | train | evaluate | all (default all).
COPY docker/entrypoint.sh /app/docker/entrypoint.sh
RUN chmod +x /app/docker/entrypoint.sh

ENTRYPOINT ["bash", "/app/docker/entrypoint.sh"]
CMD ["all"]
