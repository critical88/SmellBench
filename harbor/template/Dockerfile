FROM smellbench_{repo_name}:latest
WORKDIR /workspace/project/{repo_name}

COPY ../smell.diff /workspace/

RUN git apply /workspace/smell.diff

RUN git rev-parse HEAD > /workspace/smell_commit_id.txt

COPY ../instance.json /workspace/

CMD ["/bin/bash"]
