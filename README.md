# AoSE-GCN
A PyTorch implementation of AoSE-GCN "AoSE-GCN: Attention-aware Aggregation Operator for Spatial-Enhanced GCN".

## Abstract
<p align="justify">
Graph Neural Networks (GNNs) have shown remarkable success in various graph-relevant applications, which can be generally classified into spatial-based and spectral-based methods. Particularly, spatial methods capture local neighborhoods well but lack global structural insight since they are defined on the local nodes based on the aggregator operators. Adversely, spectral methods incorporate global structural information but struggle with local details due to the nature of the Laplacian matrix. Notably, spectral methods employ frequency component adjustment on filters to achieve effective convolutions, yet they remain less flexible compared to spatial methods. The challenge lies in balancing these approaches, enabling GNN models to capture both the global-level and local-level information, thus promoting graph representation learning. To tackle the problem, we introduce a novel attention-aware aggregation operator, denoted as $L_{att}$, by appending the attention score as additional weight in the Laplacian matrix. Inspired by its benefits, we integrate $L_{att}$ into the GCN model to perceive different levels of fields, called AoSE-GCN. Notably, our $L_{att}$ is not limited to GCN where any spectral methods can be easily plugged in. Extensive experiments on benchmark datasets validate the superiority of AoSE-GCN for node classification tasks in fully-supervised or semi-supervised settings.</p>

## Code Architecture
```
|── src                        # Source code
   └── LogerFactory            # Log saving utility class
   └── process                 # load datasets scripts
   └── cudaUtils               # GPU information display tool
   └── models                  # models
   └── layers                  # Convolution layers
|  └── models               # code for models
|── full-train              # Full-supervised classification
└── train.py                # Semi-supervised classification
```

##  Train 
```
python train.py
or
python full-train.py 
```

## Citation
```
...
```