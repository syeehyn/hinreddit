FROM ucsdets/scipy-ml-notebook

USER root

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y default-jre && \
    apt-get install -y default-jdk

COPY requirements.txt /tmp
RUN pip install --no-cache-dir -r /tmp/requirements.txt  && \
	fix-permissions $CONDA_DIR

RUN conda install -y -c pytorch \
    torchvision=0.5.0=py36_cu100 \
    && conda clean -ya
# Install HDF5 Python bindings.
RUN conda install -y h5py=2.8.0 \
    && conda clean -ya
RUN pip install h5py-cache==1.0
# Install TorchNet, a high-level framework for PyTorch.
RUN pip install torchnet==0.0.4
# Install Graphviz.
RUN conda install -y graphviz=2.40.1 python-graphviz=0.8.4 \
    && conda clean -ya
# Install OpenCV3 Python bindings.
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    libgtk2.0-0 \
    libcanberra-gtk-module \
    && sudo rm -rf /var/lib/apt/lists/*
RUN conda install -y -c menpo opencv3=3.1.0 \
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


# ARG CUDA=cu100


# RUN pip install torch-scatter==latest+$CUDA -f https://pytorch-geometric.com/whl/torch-1.4.0.html
# RUN pip install torch-sparse==latest+$CUDA -f https://pytorch-geometric.com/whl/torch-1.4.0.html
# RUN pip install torch-cluster==latest+$CUDA -f https://pytorch-geometric.com/whl/torch-1.4.0.html
# RUN pip install torch-spline-conv==latest+$CUDA -f https://pytorch-geometric.com/whl/torch-1.4.0.html
# RUN pip install torch-geometric
