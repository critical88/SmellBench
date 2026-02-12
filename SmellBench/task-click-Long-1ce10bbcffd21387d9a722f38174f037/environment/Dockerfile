FROM continuumio/miniconda3:latest
WORKDIR /workspace
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

ARG REPO_URL=None
ARG COMMIT_ID=1d038f270701498433cb432f54db89f95f07a845

RUN git clone --recursive "$REPO_URL" repo     && cd repo     && git checkout "$COMMIT_ID"

WORKDIR /workspace/repo
RUN conda create -n click-dev python==3.10 pytest pytest-cov
SHELL ["bash", "-c"]
RUN echo "source activate click-dev" >> ~/.bashrc
ENV PATH /opt/conda/envs/click-dev/bin:$PATH

RUN pip install -e ".[dev]"

CMD ["/bin/bash"]
