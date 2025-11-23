FROM nvcr.io/nvidia/pytorch:25.09-py3

# 1. System updates
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    ninja-build \
    build-essential

# 2. Upgrade pip
RUN pip install --upgrade pip && \
    pip install cmake --upgrade

# 3. Install dependencies with PINNED versions
RUN pip install \
    "transformers==4.56.0" \
    "peft>=0.17.0" \
    "trl>=0.21.0" \
    "accelerate>=1.1.1" \
    "datasets>=3.0.0" \
    "lm_eval[hf]==0.4.9.1" \
    deepspeed \
    wandb \
    bitsandbytes \
    sentencepiece
    # protobuf \
    # scipy

# 4. Install CUDA-fused xIELU
# We use --no-build-isolation to ensure it sees the PyTorch version installed in the image
ENV CUDA_HOME=/usr/local/cuda
RUN pip install "git+https://github.com/nickjbrowning/XIELU" --no-build-isolation
    
# 5. Create workspace
RUN mkdir -p /workspace
WORKDIR /workspace