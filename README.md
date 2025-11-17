# Acc-SpMM


- This repository is forked from the official implementation of Acc-SpMM: Accelerating General-purpose Sparse Matrix-Matrix Multiplication with GPU Tensor Cores, PPoPP2025.
    - [arXiv](https://arxiv.org/pdf/2501.09251)
    - [ACM Digital Library](https://dl.acm.org/doi/10.1145/3710848.3710888)

- This repository fixes the typos/issues in the official code.
    - Check the SpMM output error with cublas in `src/mma_tf32.cu`.
    - Fix all the index issues in `include/mma_tf32.cuh`.
    - Reset output for the last kernel launch in `include/tf32_comp.hpp` to obtain a single round output, since the adap kernel uses atomic add and thus cannot overwrite the output matrix C, which has been written for many times in previous rounds.
    - Align the compression format converting process in `include/format.cuh` with the bitmapping order of the SpMM kernels in `src/mma_tf32.cu`.

- This repository has been tested on RTX4090. Compared to cublas, the max absolute error per element is restricted to around 1e-3, and the total absolute error is restricted to around 1e2.

## Requirements

- Supported GPU: 
    - A800 
    - H100
    - RTX4090
- CUDA >= 11.8

## Usage

1. Clone the repository

    ```
    git clone https://github.com/AI4SClab/AccSpMM.git
    cd AccSpMM
    ```

2. Run the code

    ```
    mkdir build && cd build
    cmake .. && make
    cd ..
    ./mma $file_path $feature_dim   # you can use the matrix in folder dataset/test
    ```

## Citation

To cite this project, you can use the following BibTex citation.

```
@inproceedings{zhao2025acc,
  title={Acc-SpMM: Accelerating General-purpose Sparse Matrix-Matrix Multiplication with GPU Tensor Cores},
  author={Zhao, Haisha and Li, San and Wang, Jiaheng and Zhou, Chunbao and Wang, Jue and Xin, Zhikuang and Li, Shunde and Liang, Zhiqiang and Pan, Zhijie and Liu, Fang and others},
  booktitle={Proceedings of the 30th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming},
  pages={326--338},
  year={2025}
}
```
