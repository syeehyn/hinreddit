ARG BASE_CONTAINER=jupyter/scipy-notebook:2ce7c06a61a1
ARG DATAHUB_CONTAINER=ucsdets/datahub-base-notebook:2019.4.9

FROM $DATAHUB_CONTAINER as datahub

FROM $BASE_CONTAINER

LABEL maintainer="UC San Diego ITS/ETS <ets-consult@ucsd.edu>"

USER root

COPY --from=datahub /usr/share/datahub/scripts/* /usr/share/datahub/scripts/
RUN /usr/share/datahub/scripts/install-all.sh

#JAVA
######################################
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y default-jre && \
    apt-get install -y default-jdk

######################################
# CLI (non-conda) CUDA compilers, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get purge --autoremove -y curl && \
rm -rf /var/lib/apt/lists/*

ENV CUDA_VERSION 10.1.243

ENV CUDA_PKG_VERSION 10-1=$CUDA_VERSION-1

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-cudart-$CUDA_PKG_VERSION \
cuda-compat-10-1 && \
ln -s cuda-10.1 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

# Required for nvidia-docker v1
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.1 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411"
# Pre-generate font cache so the user does not see fc-list warning when
# importing datascience. https://github.com/matplotlib/matplotlib/issues/5836
RUN pip install --no-cache-dir datascience okpy PyQt5 && \
	python -c 'import matplotlib.pyplot' && \
	conda remove --quiet --yes --force qt pyqt || true && \
	conda clean -tipsy

###########################
# Now the ML toolkits (cuda9 until we update our Nvidia drivers)
RUN conda install -c anaconda --yes  \
                cudatoolkit=10.1 \
                cudnn nccl \
		tensorboard=2.1.0 \
		tensorflow=2.1.0 \
		tensorflow-base=2.1.0 \
		tensorflow-gpu=2.1.0 \
        && conda install -c pytorch --yes \
                pytorch=1.4.0 \
                torchvision=0.5.0 \
        && conda install --yes \
                nltk spacy \
        && conda clean -afy && fix-permissions $CONDA_DIR

# Install tensorboard plugin for Jupyter notebooks
RUN pip install --no-cache-dir jupyter-tensorboard && \
	jupyter tensorboard enable --sys-prefix
#Additional
COPY requirements.txt /tmp
RUN pip install --no-cache-dir -r /tmp/requirements.txt  && \
	fix-permissions $CONDA_DIR

# Install HDF5 Python bindings.
RUN conda install -y h5py=2.8.0 \
    && conda clean -ya
RUN pip install h5py-cache==1.0
# Install TorchNet, a high-level framework for PyTorch.
RUN pip install torchnet==0.0.4
# Install Graphviz.
RUN conda install -y graphviz=2.40.1 python-graphviz=0.8.4 \
    && conda clean -ya
# Install PyTorch Geometric.
RUN CPATH=/usr/local/cuda/include:$CPATH \
    && LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    && DYLD_LIBRARY_PATH=/usr/local/cuda/lib:$DYLD_LIBRARY_PATH

RUN pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html \
    && pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html \
    && pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html \
    && pip install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html \
    && pip install torch-geometric

RUN conda install -y tsnecuda cuda101 -c cannylab
RUN conda install -y -c dglteam dgl-cuda10.0
RUN conda install -y -c stellargraph stellargraph


# Spark dependencies
ENV APACHE_SPARK_VERSION=2.4.5 \
    HADOOP_VERSION=2.7

RUN apt-get -y update && \
    apt-get install --no-install-recommends -y openjdk-8-jre-headless ca-certificates-java && \
    rm -rf /var/lib/apt/lists/*

# Using the preferred mirror to download the file
RUN cd /tmp && \
    wget -q $(wget -qO- https://www.apache.org/dyn/closer.lua/spark/spark-${APACHE_SPARK_VERSION}/spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz\?as_json | \
    python -c "import sys, json; content=json.load(sys.stdin); print(content['preferred']+content['path_info'])") && \
    echo "2426a20c548bdfc07df288cd1d18d1da6b3189d0b78dee76fa034c52a4e02895f0ad460720c526f163ba63a17efae4764c46a1cd8f9b04c60f9937a554db85d2 *spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" | sha512sum -c - && \
    tar xzf spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz -C /usr/local --owner root --group root --no-same-owner && \
    rm spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz
RUN cd /usr/local && ln -s spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} spark


# Spark and Mesos config
ENV SPARK_HOME=/usr/local/spark
ENV PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.7-src.zip \
    SPARK_OPTS="--driver-java-options=-Xms1024M --driver-java-options=-Xmx4096M --driver-java-options=-Dlog4j.logLevel=info" \
    PATH=$PATH:$SPARK_HOME/bin

ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64

# Install pyarrow
RUN conda install --quiet -y 'pyarrow' && \
    conda clean --all -f -y && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER
