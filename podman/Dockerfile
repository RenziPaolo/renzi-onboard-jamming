# Use the official CachyOS Docker image as the base
FROM docker.io/cachyos/cachyos-v3:latest

# Update the system and install essential programs
RUN pacman -Syu --noconfirm && \
    pacman -S --noconfirm \
    python \
    python-pip \
    base-devel \
    git \
    wget \
    cachyos-rate-mirrors \
    bash \
    tar \
    bzip2 \
    xz
   
RUN cachyos-rate-mirrors -k && \
    pacman -Syu --noconfirm && \
    pacman -S --noconfirm \
    cuda \
    cudnn 

# Download Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    # Run the installer silently
    bash ~/miniconda.sh -b -p /opt/conda && \
    # Clean up the installer
    rm ~/miniconda.sh
    
RUN /opt/conda/bin/conda clean -afy && \
    rm -rf /opt/conda/pkgs/*

ENV PATH="/opt/conda/bin:$PATH"

RUN conda --version

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,video
 
# Set the working directory
WORKDIR /app

# Copy your application code (optional)
COPY . /app
COPY requirements.txt /
# Create and set permissions for the pip cache directory
RUN mkdir /.cache && chmod 777 /.cache

RUN python -m venv .venv

# Install essential packages: torch, torchvision, torchaudio
RUN . .venv/bin/activate && \
    pip install --upgrade pip

# Copy the requirements file and install dependencies

RUN . .venv/bin/activate && pip install -r /requirements.txt

RUN . .venv/bin/activate && pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Declare the volume (optional, for documentation)
VOLUME /mnt/ssd2/renzi
VOLUME /home/renzi
