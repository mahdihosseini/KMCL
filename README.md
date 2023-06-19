# End-to-End Supervised Multilabel Contrastive Learning

Official PyTorch Implementation

> Ahmad Sajedi*, Samir Khaki*, Konstantinos N. Plataniotis, Mahdi S. Hosseini
> * : equal contribution

**Abstract**

Multilabel representation learning is recognized as a challenging problem that can be associated with either label dependencies between object categories or data related issues such as the inherent imbalance of positive/negative samples. Recent advances address these challenges from model- and data-centric viewpoints. In model-centric, the label correlation is obtained by an external model designs (e.g., graph CNN) to incorporate an inductive bias for training. However, they fail to design an end-to-end training framework, leading to high computational complexity. On the contrary, in data-centric, the realistic nature of the dataset is considered for improving the classification while ignoring the label dependencies. In this paper, we propose a new end-to-end training framework–dubbed KMCL (Kernel-based Mutlilabel Contrastive Learning)–to address the shortcomings of both model- and data-centric designs. The KMCL first transforms the embedded features into a mixture of exponential kernels in Gaussian RKHS. It is then followed by encoding an objective loss that is comprised of (a) reconstruction loss to reconstruct kernel representation, (b) asymmetric classification loss to address the inherent imbalance problem, and (c) contrastive loss to capture label correlation. The KMCL models the uncertainty of the feature encoder while maintaining a low computational footprint. Extensive experiments are conducted on both computer vision and medical image classification tasks to showcase the consistent improvements of KMCL over the SOTA methods.


## Kernel-based Mutlilabel Contrastive Learning (KMCL) Implementation
In this PyTorch [file](\src\loss_functions\losses.py), we provide 
the KMCL implementations which augments the base ASL loss function. 

For the multi-label case (sigmoids), the implementation is : 
- ```class KMCL_Loss(nn.Module)```

This class leverages three separate loss formulations:
- ```ASL``'
- ```Reconstruction Loss```
- ```KMCL Loss```
The losses and generic class take keyword arguments to specify the loss case (either "isotropic" or "anisotropic") as well as the distance measure (Battacharya, Mahalonbis, RBF).


Running either [train.py](\train.py) or [train_dist.py](\train_dist.py) will execute the end-to-end contrastive learning for the dataset of interest. The latter runs the script under a fully distributed framework enabling our method to scale easily to larger image resolutions, architectures, and batch size configurations.

This project source code is based on https://github.com/Alibaba-MIIL/ASL.
