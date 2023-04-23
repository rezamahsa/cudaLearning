# cudaLearning Abstract
from : https://github.com/brucefan1983/CUDA-Programming

## 1.1 Introduction to GPU 

* Tesla series: good for scientific computing but expensive.
* GeForce series: cheaper but less professional. 
* Quadro series: kind of between the above two.
* Jetson series: embedded device 

Every GPU has a version number `X.Y` to indicate its **compute capability**.
| Major compute capability  | architecture name |   release year  |
|:------------|:---------------|:--------------|
| `X=1` | Tesla | 2006 |
| `X=2` | Fermi | 2010 |
| `X=3` | Kepler | 2012 |
| `X=5` | Maxwell | 2014 |
| `X=6` | Pascal | 2016 |
| `X=7` | Volta | 2017 |
| `X.Y=7.5` | Turing | 2018 |
| `X=8` | Ampere | 2020 |

GPUs older than Pascal will become deprecated soon
performance of a GeForce GPU is only 1/32 of its single-precision performance

## 1.2 Introduction to CUDA 

| CUDA versions | supported GPUs |
|:------------|:---------------|
|CUDA 11.0 |  Compute capability 3.5-8.0 (Kepler to Ampere) |
|CUDA 10.0-10.2 | Compute capability 3.0-7.5 (Kepler to Turing) |
|CUDA 9.0-9.2 | Compute capability 3.0-7.2  (Kepler to Volta) | 
|CUDA 8.0     | Compute capability 2.0-6.2  (Fermi to Pascal) | 

## 1.3 Installing CUDA 

For Linux, check this manual: https://docs.nvidia.com/cuda/cuda-installation-guide-linux

## 2.1 Hello CUDA 

```c++
__global__ void hello_from_gpu()
{
    printf("Hello World from the GPU!\n");
}    
```
The order of the qualifiers, `__global__` and `void`, are not important. That is, we can also write the kernel as:
```c++
void __global__ hello_from_gpu()
{
    printf("Hello World from the GPU!\n");
}
```

```shell
$ nvcc hello1.cu
```
## 2.1 Hello CUDA 2

```c++
#include <stdio.h>
__global__ void hello_from_gpu()
{
    printf("Hello World from the GPU!\n");
}
int main(void)
{
    hello_from_gpu<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
```
