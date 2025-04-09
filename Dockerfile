# Use a CUDA-enabled base image
FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

# Set the working directory
WORKDIR /app

# Install necessary dependencies
RUN apt-get update && apt-get install -y wget curl git

# Install conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -b -p /opt/conda
ENV PATH="/opt/conda/bin:$PATH"

# Copy the environment.yml file and create the conda environment
COPY environment.yml .
RUN conda env create -f environment.yml

# Activate the conda environment
ENV CONDA_DEFAULT_ENV=base
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]

# Copy the detection and segmentation directories
COPY detection detection
COPY segmentation segmentation

# # Install detection package
# RUN cd detection/ && pip install -e -v .

# # Install segmentation package
# RUN cd segmentation/ && pip install -e -v .

# Define the entry point
CMD ["/bin/bash"]
