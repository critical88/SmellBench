FROM continuumio/miniconda3:latest
WORKDIR /workspace
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

ARG REPO_URL=None
ARG COMMIT_ID=1392cbe3c79cdb93f9282747841d648770f60249

RUN git clone --recursive "$REPO_URL" repo     && cd repo     && git checkout "$COMMIT_ID"

WORKDIR /workspace/repo
RUN conda env create --file environment.yml
SHELL ["bash", "-c"]
RUN echo "source activate mpl-dev" >> ~/.bashrc
ENV PATH /opt/conda/envs/mpl-dev/bin:$PATH

RUN ['python -m pip install --verbose --no-build-isolation --editable ".[dev]"']

CMD ["/bin/bash"]
