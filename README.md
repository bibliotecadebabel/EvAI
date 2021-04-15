## Background

In this repository, we present new neural architecture search algorithms used to obtain this paper's numerical results:

[Traditional and accelerated gradient descent for neural architecture search, (J. Morales, F. Morales, and N. Trillos), arXiv:2006.15218 (2020).](https://arxiv.org/abs/2006.15218)

This paper proposes a new family of algorithms for neural architecture search derived from a new geometrical structure induced by the optimal transport problem on semi-discrete space. This structure was introduced in the paper:  

[Semi-discrete optimization through semi-discrete optimal transport: a framework for neural architecture search, (J. Morales, N. Trillos), arXiv:2006.15221 (2020).](https://arxiv.org/abs/2006.15221)

## To Use
To start an architecture search, follow these steps:
Execute run_product.py
select the number of threads, max layers, set mutation coefficient (1 is suggested), and number children architectures (4-8).
The best architecture is saved every time number of layers changes.
To train final architecture, stop and execute post_training: execute run_post_training.py.

