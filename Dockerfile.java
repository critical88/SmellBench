FROM continuumio/miniconda3:latest

WORKDIR /workspace/

# Install system dependencies for Java projects
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    openjdk-17-jdk \
    maven \
    && rm -rf /var/lib/apt/lists/*

# Verify Java and Maven installation
RUN java -version && mvn -version

# Set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

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
