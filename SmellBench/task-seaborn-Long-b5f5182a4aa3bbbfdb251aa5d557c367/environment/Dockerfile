FROM continuumio/miniconda3:latest
WORKDIR /workspace
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

ARG REPO_URL=None
ARG COMMIT_ID=d00a27df750130e28e4801caaedc782433863a37

RUN git clone --recursive "$REPO_URL" repo     && cd repo     && git checkout "$COMMIT_ID"

WORKDIR /workspace/repo
RUN conda create -n seaborn-dev python==3.10
SHELL ["bash", "-c"]
RUN echo "source activate seaborn-dev" >> ~/.bashrc
ENV PATH /opt/conda/envs/seaborn-dev/bin:$PATH

RUN ['pip install flit_core click pytest==8.2', 'python -m pip install --verbose --no-build-isolation --editable ".[dev]"']

CMD ["/bin/bash"]
