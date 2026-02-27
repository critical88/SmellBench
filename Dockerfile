FROM continuumio/miniconda3:latest
WORKDIR /workspace/
RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*

RUN conda create -n testbed python==3.12
COPY ./requirements.txt /workspace
COPY ./eval_requirements.txt /workspace

RUN conda run -n testbed pip install -r eval_requirements.txt

RUN echo "source activate testbed" >> ~/.bashrc
ENV PATH /opt/conda/envs/testbed/bin:$PATH

CMD ["/bin/bash"]