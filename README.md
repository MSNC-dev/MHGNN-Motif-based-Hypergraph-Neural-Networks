
# MHGNN

The source codes and datasets for `Exploring High-order Correlations With Motif-based Hypergraph Neural Networks forEpidemic Forecasting`. Specifically, the codes are in the `\src`, while data is in the `\data`.
Specifically, the source codes are provided in the `src` directory, the datasets are stored in the `data` directory, and the constructed motif networks are included in the `motif` directory.


## 1. Introduction

Accurate epidemic forecasting is critical for timely public health responses and optimal control measures.
Deep learning methods which utilize recurrent neural networks (RNNs) capable of sequence modeling are explored recently. 
Later on, Graph Neural Networks (GNNs) are adopted to further capture both local and global dependencies of regions by representing regions and their pairwise relations as a graph. 
However, in real-world scenarios the complex transmission patterns are often high-order. For instance, there exists infection dependencies of surrounding areas as epidemics propagates in all directions. Also, the infection counts of two regions with more common neighboring regions are more similar. 
To address these limitations, we propose a motif-based Hypergraph Neural Networks(MHGNN) framework integrating hypergraphs and network motifs to model high-order dependencies. On the one hand, network motifs are utilized to eliminate insignificant relations to identify critical message-passing pathways. On the other hand, hyperedges are designed to capture the underlying high-order dependencies of groups of similar region nodes exploring both local geographical adjacency and global correlations of non-adjacent regions. 
Experimental results demonstrate that our proposed model improves forecasting accuracy on several public datasets compared with state-of-the-art methods, showing the effectiveness of the model. 

## 2. Datasets
### 2.1 Epidemic Statistics

The Influenza-related datasets are released by [Cola-GNN](https://github.com/amy-deng/colagnn) and the COVID-related data is publicly avaliable at [JHU-CSSE](https://github.com/CSSEGISandData/COVID-19).

### 2.2 Motif Networks

The motif networks are generated using PGD. We first download the [PGD](https://github.com/nkahmed/PGD) source code and fix bug according to issue. Then, the graph file in .mtx format is used as the input to PGD. For example, the following command can be used to generate the corresponding .micro file:

./pgd -f sample_graph.mtx --micro sample_graph.micro

The generated .micro files are placed in the motif directory and used as the corresponding motif networks. Each .micro file should have the same filename as its corresponding adjacency matrix file.

The input .mtx file follows an edge-list format. The first line records the number of nodes and edges:
```text
num_nodes num_nodes num_edges
```

The following lines contain the indices of the two nodes connected by edge:
```text
source_node target_node
```

## 3. Quick Start

All programs are implemented using Python 3.11.4 and PyTorch 2.0.1 with CUDA 12.2 (2.0.1 cu122) in an Ubuntu server with an NVIDIA GeForce RTX 3090 GPU.

```shell
cd MHGNN
pip install -r requirements.txt
```

run the US-State dataset as example:
```shell
python src/train.py --cuda --gpu 0 --lr 0.005 --horizon 5 --khop_num 2 --KNNhop --data state360 --sim_mat state-adj-49 
```
### 3.1 Parameters
+ *dataset*: time series data.
+ *sim_mat*: the adjacent matrix.
+ *lr*: learning rate.
+ *batch*: batch size.
+ *epoch*: the number of epochs of traning process.
+ *patience*: we conduct early stop with fixed patience.
+ *k*: dimension = k * 10.
+ *n*: number of hypergraph convolution layers.
+ *window*: length of the historical observation window.
+ *KNNhop*: choose include the KNN-based hypergraph branch or not.
+ *knn_nodes*: Number of nearest neighbors in KNN hyperedge branch. If set to -1, use log2(num_nodes) as default.
+ *khop_num*: Maximum order of k-hop hypergraphs to construct.

## More about EPIDEMICs

+ Seasonal influenza: [https://www.who.int/en/news-room/fact-sheets/detail/influenza-(seasonal)](https://www.who.int/en/news-room/fact-sheets/detail/influenza-(seasonal))
+ COVID-19 pandemic: [https://covid19.who.int/](https://covid19.who.int/)
+ Global statistics: [https://clustrmaps.com/coronavirus/](https://clustrmaps.com/coronavirus/)
+ The epidemic surveillance system of Lanzhou University: [http://covid-19.lzu.edu.cn/index.htm](http://covid-19.lzu.edu.cn/index.htm)
+ HHS region: [https://www.hhs.gov/about/agencies/iea/regional-offices/index.html](https://www.hhs.gov/about/agencies/iea/regional-offices/index.html)


