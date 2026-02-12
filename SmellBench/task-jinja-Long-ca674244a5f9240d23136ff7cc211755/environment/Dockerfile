FROM continuumio/miniconda3:latest
WORKDIR /workspace
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

ARG REPO_URL=None
ARG COMMIT_ID=15206881c006c79667fe5154fe80c01c65410679

RUN git clone --recursive "$REPO_URL" repo     && cd repo     && git checkout "$COMMIT_ID"

WORKDIR /workspace/repo
RUN conda create -n jinja-dev python==3.10 pytest pytest-cov trio
SHELL ["bash", "-c"]
RUN echo "source activate jinja-dev" >> ~/.bashrc
ENV PATH /opt/conda/envs/jinja-dev/bin:$PATH

RUN pip install -e .

CMD ["/bin/bash"]
