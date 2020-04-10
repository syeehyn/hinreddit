FROM ucsdets/scipy-ml-notebook

USER root

ENV CUDA cu100

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y default-jre && \
    apt-get install -y default-jdk

RUN conda install --yes \
    pyspark

COPY requirements.txt /tmp
RUN pip install --no-cache-dir -r /tmp/requirements.txt  && \
	fix-permissions $CONDA_DIR

RUN pip install torch-scatter==latest+$CUDA -f https://pytorch-geometric.com/whl/torch-1.4.0.html
RUN pip install torch-sparse==latest+$CUDA -f https://pytorch-geometric.com/whl/torch-1.4.0.html
RUN pip install torch-cluster==latest+$CUDA -f https://pytorch-geometric.com/whl/torch-1.4.0.html
RUN pip install torch-spline-conv==latest+$CUDA -f https://pytorch-geometric.com/whl/torch-1.4.0.html
RUN pip install torch-geometric

RUN apt-get -y update && \
    apt-get install --no-install-recommends -y openjdk-8-jre-headless ca-certificates-java && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64