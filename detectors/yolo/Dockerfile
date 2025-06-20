# syntax=docker/dockerfile:1.13-labs
FROM debian:bookworm-slim AS builder
# Install GIT
RUN apt-get update && apt-get install -y \
    git libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

FROM --platform=$TARGETPLATFORM public.ecr.aws/clarifai-models/torch:2.5.1-py3.11-cu124-b8ae56bf3b7c95e686ca002b07ca83d259c716eb as final

# Copy git binary and required libraries
COPY --from=builder /usr/bin/git /usr/bin/git
COPY --from=builder /lib /lib
COPY --from=builder /lib64 /lib64
COPY --from=builder /usr/lib /usr/lib

COPY --link requirements.txt /home/nonroot/requirements.txt

# Update clarifai package so we always have latest protocol to the API. Everything should land in /venv
RUN ["pip", "install", "--no-cache-dir", "-r", "/home/nonroot/requirements.txt"]
RUN ["pip", "show", "clarifai"]

# Set the NUMBA cache dir to /tmp
# Set the TORCHINDUCTOR cache dir to /tmp
# The CLARIFAI* will be set by the templaing system.
ENV NUMBA_CACHE_DIR=/tmp/numba_cache \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache \
    HOME=/tmp \
    DEBIAN_FRONTEND=noninteractive

#####
# Copy the files needed to download
#####
# This creates the directory that HF downloader will populate and with nonroot:nonroot permissions up.
COPY --chown=nonroot:nonroot downloader/unused.yaml /home/nonroot/main/1/checkpoints/.cache/unused.yaml

#####
# Download checkpoints if config.yaml has checkpoints.when = "build"
COPY --link=true config.yaml /home/nonroot/main/
#RUN ["python", "-m", "clarifai.cli", "model", "download-checkpoints", "/home/nonroot/main", "--out_path", "/home/nonroot/main/1/checkpoints", "--stage", "build"]
#####

# Copy in the actual files like config.yaml, requirements.txt, and most importantly 1/model.py
# for the actual model.
# If checkpoints aren't downloaded since a checkpoints: block is not provided, then they will
# be in the build context and copied here as well.
COPY --link=true 1 /home/nonroot/main/1
# At this point we only need these for validation in the SDK.
COPY --link=true requirements.txt config.yaml /home/nonroot/main/

# Add the model directory to the python path.
ENV PYTHONPATH=${PYTHONPATH}:/home/nonroot/main \
    CLARIFAI_PAT=${CLARIFAI_PAT} \
    CLARIFAI_USER_ID=${CLARIFAI_USER_ID} \
    CLARIFAI_RUNNER_ID=${CLARIFAI_RUNNER_ID} \
    CLARIFAI_NODEPOOL_ID=${CLARIFAI_NODEPOOL_ID} \
    CLARIFAI_COMPUTE_CLUSTER_ID=${CLARIFAI_COMPUTE_CLUSTER_ID} \
    CLARIFAI_API_BASE=${CLARIFAI_API_BASE:-https://api.clarifai.com}

# Finally run the clarifai entrypoint to start the runner loop and local dev server.
# Note(zeiler): we may want to make this a clarifai CLI call.
ENTRYPOINT ["python", "-m", "clarifai.runners.server"]
CMD ["--model_path", "/home/nonroot/main"]
#############################
