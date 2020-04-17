FROM ucsdets/scipy-ml-notebook

USER root

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y default-jre && \
    apt-get install -y default-jdk

RUN conda install -y -c pytorch \
    pytorch=1.4.0 \
    torchvision=0.5.0

RUN conda install -c anaconda --yes  \
    tensorboard=2.2.0 \
    tensorflow=2.2.0 \
    tensorflow-base=2.2.0 \
    tensorflow-gpu=2.2.0 \
    numpy=1.17

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

RUN pip install torch-scatter==latest+cu100 -f https://pytorch-geometric.com/whl/torch-1.4.0.html \
    && pip install torch-sparse==latest+cu100 -f https://pytorch-geometric.com/whl/torch-1.4.0.html \
    && pip install torch-cluster==latest+cu100 -f https://pytorch-geometric.com/whl/torch-1.4.0.html \
    && pip install torch-spline-conv==latest+cu100 -f https://pytorch-geometric.com/whl/torch-1.4.0.html \
    && pip install torch-geometric
# Install Pyspark
RUN conda install --yes \
    pyspark
RUN apt-get -y update && \
    apt-get install --no-install-recommends -y openjdk-8-jre-headless ca-certificates-java && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64
