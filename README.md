# Hinreddit

Project|build Status
---|---
ETL | Finished
Labeling | Finished
Feature Extraction| Finished
ML Deployment | In Progress

[Development Timeline](./writeups/DEVTIMELINE.md)

This project aimes to investigate contents from Reddit by detecting positive or negative comments with technique of Heterogeneous Information Network rather than traditional approach of test processing.

[Overview](./writeups/OVERVIEW.md)

- [Hinreddit](#hinreddit)
  - [Getting Started](#getting-started)
    - [Prerequisite](#prerequisite)
      - [Use Dockerfile](#use-dockerfile)
    - [Usage](#usage)
  - [Description of Contents](#description-of-contents)
  - [Contribution](#contribution)
    - [Authors](#authors)
    - [Advisors](#advisors)
  - [References](#references)
  - [License](#license)

----

## Getting Started

### Prerequisite

The project is mainly built upon following packages:

- Data Preprocessing & Feature Extraction

  - [Pandas](https://pandas.pydata.org/)

  - [Spark](https://spark.apache.org/)

- Labeling & ML Deployment

  - [Pytorch = 1.4](https://pytorch.org/)
  
  - [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric)

  - [Fast-Bert](https://github.com/kaushaltrivedi/fast-bert)

#### Use Dockerfile

  You can build a docker image out of the provided [DockerFile](Dockerfile)

  ```bash
  $ docker build . # This will build using the same env as in a)
  ```

  Run a container, replacing the ID with the output of the previous command

  ```
  $ docker run -it -p 8888:8888 -p 8787:8787 <container_id_or_tag>
  ```
  
  The above command will give an URL (Like http://(container_id or 127.0.0.1):8888/?token=<sometoken>) which can be used to access the notebook from browser. You may need to replace the given hostname with "localhost" or "127.0.0.1".

### Usage

<!-- TODO --> 

----

## Description of Contents

``` bash
├── LICENSE
├── README.md
├── config
├── notebooks
├── src
└── tests
```

----

## Contribution

### Authors

- [Chengyu Chen](https://github.com/anniechen0127)
- [Yu-chun Chen](https://github.com/yuc330)
- [Yanyu Tao](https://github.com/lilytaoyy)
- [Shuibenyang Yuan](https://github.com/shy166)

### Advisors

- [Aaron Fraenkel](https://afraenkel.github.io/)
- [Shivam Lakhotia](https://github.com/shivamlakhotia)

<sup>Authors contributed equally to this project</sup>

----

## References

``` 
@paper{Hou/Ye/2017,
  title={HinDroid: {An Intelligent Android Malware Detection System Based on Structured Heterogeneous Information Network}},
  author={Hou, Ye, Song, Abdulhayoglu}
  year={2017}
}
@inproceedings{Fey/Lenssen/2019,
  title={Fast Graph Representation Learning with {PyTorch Geometric}},
  author={Fey, Matthias and Lenssen, Jan E.},
  booktitle={ICLR Workshop on Representation Learning on Graphs and Manifolds},
  year={2019},
}
@article{turc2019,
  title={Well-Read Students Learn Better: On the Importance of Pre-training Compact Models},
  author={Turc, Iulia and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1908.08962v2 },
  year={2019}
}
@course{Koutra/2018,
  title={Mining Large-scale Graph Data},
  author={Danai Koutra},
  link={http://web.eecs.umich.edu/~dkoutra/courses/W18_598/},
  year={2018}
}
@collection{src-d/2019,
  title={Awesome Machine Learning On Source Code},
  author={src-d},
  link={https://github.com/src-d/awesome-machine-learning-on-source-code},
  year={2019}
}
```

----

## License

[Apache License 2.0](LICENSE)
