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

## Hello CUDA 

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
## Hello CUDA 2

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

`cudaDeviceSynchronize();` synchronize the host and the device, making sure that the output stream for the printf function has been flushed before returning from the kernel to the host.

## A CUDA kernel using multiple threads

```c++
#include <stdio.h>
__global__ void hello_from_gpu()
{
    printf("Hello World from the GPU!\n");
}
int main(void)
{
    hello_from_gpu<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}

```
    Hello World from the GPU!
    Hello World from the GPU!
    Hello World from the GPU!
    Hello World from the GPU!
    Hello World from the GPU!
    Hello World from the GPU!
    Hello World from the GPU!
    Hello World from the GPU!
```

## Using thread indices in a CUDA kernel

```c++
#include <stdio.h>
__global__ void hello_from_gpu()
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    printf("Hello World from block %d and thread %d!\n", bid, tid);
}
int main(void)
{
    hello_from_gpu<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

```
    Hello World from block 1 and thread 0.
    Hello World from block 1 and thread 1.
    Hello World from block 1 and thread 2.
    Hello World from block 1 and thread 3.
    Hello World from block 0 and thread 0.
    Hello World from block 0 and thread 1.
    Hello World from block 0 and thread 2.
    Hello World from block 0 and thread 3.
```
and sometimes we get the following output,
```
    Hello World from block 0 and thread 0.
    Hello World from block 0 and thread 1.
    Hello World from block 0 and thread 2.
    Hello World from block 0 and thread 3.
    Hello World from block 1 and thread 0.
    Hello World from block 1 and thread 1.
    Hello World from block 1 and thread 2.
    Hello World from block 1 and thread 3.
```

## Generalization to multi-dimensional grids and blocks

* `blockIdx` and `threadIdx` are of type `uint3`, which is defined in `vector_types.h` as:
```c++
    struct __device_builtin__ uint3
    {
        unsigned int x, y, z;
    };    
    typedef __device_builtin__ struct uint3 uint3;
```
We can use the constructors of the struct `dim3` to define multi-dimensional grids and blocks:
```c++
    dim3 grid_size(Gx, Gy, Gz);
    dim3 block_size(Bx, By, Bz);
```
If the size of the `z` dimension is 1, we can simplify the above definitions to:
```c++
    dim3 grid_size(Gx, Gy);
    dim3 block_size(Bx, By);
```

example :
```c++
#include <stdio.h>
__global__ void hello_from_gpu()
{
    const int b = blockIdx.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    printf("Hello World from block-%d and thread-(%d, %d)!\n", b, tx, ty);
}
int main(void)
{
    const dim3 block_size(2, 4);
    hello_from_gpu<<<1, block_size>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

```
    Hello World from block-0 and thread-(0, 0)!
    Hello World from block-0 and thread-(1, 0)!
    Hello World from block-0 and thread-(0, 1)!
    Hello World from block-0 and thread-(1, 1)!
    Hello World from block-0 and thread-(0, 2)!
    Hello World from block-0 and thread-(1, 2)!
    Hello World from block-0 and thread-(0, 3)!
    Hello World from block-0 and thread-(1, 3)!
```

In general:
```c++
    int tid = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
```

## Limits on the grid and block sizes

For all the GPUs starting from the Kepler architecture, the grid size is limited to 
```c++
  gridDim.x <= 2^{31}-1
  gridDim.y <= 2^{16}-1 = 65535
  gridDim.z <= 2^{16}-1 = 65535
```
and the block size is limited to
```c++
  blockDim.x <= 1024
  blockDim.y <= 1024
  blockDim.z <= 64
```
Besides this, there is an important limit on the following product:
```c++
  blockDim.x * blockDim.y * blockDim.z <= 1024
```
**It is important to remember the above limits.**

## 2.5.2 Some important flags for `nvcc` 

The CUDA compiler driver `nvcc` first separates the source code into host code and device code. The host code will be compiled by a host C++ compiler such as `cl.exe` or `g++`. `nvcc` will first compile the device code into an intermediate PTX（Parallel Thread eXecution）code, and then compile the PTX code into a **cubin** binary. 

flag `-arch=compute_XY` to `nvcc` is needed to specify the compute capability of a **virtual architecture**

flag `-code=sm_ZW` is needed to specify the compute capability of a **real architecture**

**The compute capability of the real architecture must be no less than that of the virtual architecture.**

For example, 
```
$ nvcc -arch=compute_60 -code=sm_70 xxx.cu
```
is ok, but 
```
$ nvcc -arch=compute_70 -code=sm_60 xxx.cu
```
will result in errors.
