FROM smellbench_{REPO_NAME}:latest
WORKDIR /workspace/project/{REPO_NAME}

RUN git apply smell.diff

CMD ["/bin/bash"]
