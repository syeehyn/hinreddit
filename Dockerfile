FROM ucsdets/scipy-ml-notebook

USER root

RUN conda install --yes \
    pyspark

RUN pip --no-cache-dir install \
    multiprocess \
    beautifulsoup4 \
    pip install praw \
    lxml \
    pyarrow \
    sqlalchemy \
    scipy \
    pathos \
    p_tqdm


RUN apt-get -y update && \
    apt-get install --no-install-recommends -y openjdk-8-jre-headless ca-certificates-java && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64