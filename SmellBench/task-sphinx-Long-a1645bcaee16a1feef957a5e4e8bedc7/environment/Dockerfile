FROM continuumio/miniconda3:latest
WORKDIR /workspace
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

ARG REPO_URL=None
ARG COMMIT_ID=8ddba7e4cb5fdc035b9b18b2a8594724a3a33b86

RUN git clone --recursive "$REPO_URL" repo     && cd repo     && git checkout "$COMMIT_ID"

WORKDIR /workspace/repo
RUN conda create -n sphinx-dev -c conda-forge python==3.12 pytest pytest-cov defusedxml
SHELL ["bash", "-c"]
RUN echo "source activate sphinx-dev" >> ~/.bashrc
ENV PATH /opt/conda/envs/sphinx-dev/bin:$PATH

RUN pip install -e .

CMD ["/bin/bash"]
