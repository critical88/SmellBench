FROM continuumio/miniconda3:latest

WORKDIR /workspace/

# Install system dependencies for Go projects
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Go (version 1.21)
RUN wget https://go.dev/dl/go1.21.0.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go1.21.0.linux-amd64.tar.gz && \
    rm go1.21.0.linux-amd64.tar.gz

# Set up Go environment
ENV PATH="/usr/local/go/bin:${PATH}"
ENV GOPATH="/workspace/go"
ENV PATH="${GOPATH}/bin:${PATH}"

# Verify Go installation
RUN go version

# Create Python testbed environment (for evaluation scripts)
RUN conda create -n testbed python==3.12

# Copy requirements files
COPY ./requirements.txt /workspace
COPY ./eval_requirements.txt /workspace

# Install Python dependencies (for evaluation framework)
RUN conda run -n testbed pip install -r eval_requirements.txt

# Set up shell environment
RUN echo "source activate testbed" >> ~/.bashrc
ENV PATH /opt/conda/envs/testbed/bin:$PATH

CMD ["/bin/bash"]
