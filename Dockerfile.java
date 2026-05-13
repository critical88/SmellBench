FROM continuumio/miniconda3:latest

WORKDIR /workspace/

# Install system dependencies for Java projects
RUN apt-get update && apt-get install -y \
    git \
    build-essential 
    
RUN apt-get install -y wget \
    curl \
    maven
RUN rm -rf /var/lib/apt/lists/*
    

# Install JDK 17 from Adoptium (Temurin)
RUN wget https://download.java.net/java/GA/jdk17.0.2/dfd4a8d0985749f896bed50d7138ee7f/8/GPL/openjdk-17.0.2_linux-x64_bin.tar.gz \
    && tar -xzf openjdk-17.0.2_linux-x64_bin.tar.gz -C /opt \
    && rm openjdk-17.0.2_linux-x64_bin.tar.gz


# Set JAVA_HOME
ENV JAVA_HOME=/opt/jdk-17.0.2
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Verify Java and Maven installation
RUN java -version && mvn -version

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
