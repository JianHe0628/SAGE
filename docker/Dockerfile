FROM nvidia/cuda:12.8.0-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip git \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -m -u 1000 sage
USER sage
ENV PATH="/home/sage/.local/bin:$PATH"

# Python dependencies
WORKDIR /app
COPY --chown=sage:sage requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt \
 && python -m spacy download de_core_news_lg \
 && python -m spacy download zh_core_web_sm

# Project code
COPY --chown=sage:sage . .

# Build from project root:
#   docker build -f docker/Dockerfile -t sage .
#
# Run with datasets mounted:
#   docker run --gpus all \
#     -v /path/to/phoenix/frames:/data/phoenix \
#     -v /path/to/support_files:/app/support_files \
#     sage:latest
CMD ["bash"]
