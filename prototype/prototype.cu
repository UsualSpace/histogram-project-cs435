/*
Compile with `nvcc -arch=sm_60 prototype.cu`

input file can be created with something like `dd bs=1024 count=1 if=/dev/urandom of=./rndfile`

then `./a.out rndfile 32 32 0 20` for a 32x32 image file, in CPU mode, run 20x

There is no bounds checking so beware

The output shows 256 buckets with their counts, they should be around size/256 each

The idea was to make a Mode (the last parameter) for each example from the book, e.g., plain CPU is mode 0,
histogram_kernel_6 is called in mode 1, histogram_kernel_9 is called in mode 2, etc. but this is just some thing to get
started with.

 */

#include <cstdio>
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <stdio.h>

enum Modes { CPU = 0, GPU_SIMPLE };
static const int NUM_BINS = 256;
static const int COARSE_FACTOR = 3; // TODO:

#define CUDACHECK(err)                                                                                                 \
    do {                                                                                                               \
        cuda_check((err), __FILE__, __LINE__);                                                                         \
    } while (false)
inline void cuda_check(cudaError_t error_code, const char* file, int line) {
    if (error_code != cudaSuccess) {
        fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n", error_code, cudaGetErrorString(error_code),
                file, line);
        fflush(stderr);
        exit(error_code);
    }
}

void print_bins(unsigned int* bins) {
    for (int i = 0; i < NUM_BINS; i++) {
        if (bins[i] != 0) {
            printf("%d: %d; ", i, bins[i]);
        }
    }
    printf("\n");
}

/* Fig 9.2
    your standard single threaded CPU solution
*/
void histogram_sequential(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {

    for (unsigned int i = 0; i < width * height; ++i) {
        unsigned char b = image[i];
        ++bins[b];
    }
}

/* Fig 9.6
    introduction to primitives, this is slow but guaranteed to be correct
    talk about latency and memory bandwidth
*/
__global__ void histogram_kernel_6(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width * height) {
        unsigned char b = image[i];

        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bins_ref(bins[b]);
        bins_ref.fetch_add(1, cuda::memory_order_relaxed);
    }
}

/* Fig 9.9
 */
__global__ void histogram_kernel_9(unsigned char* image, unsigned int* bins, unsigned int* bins_pool,
                                   unsigned int width, unsigned int height) {
    unsigned int* bins_priv = &bins_pool[blockIdx.x * NUM_BINS];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width * height) {
        unsigned char b = image[i];
        cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bins_priv_ref(bins_priv[b]);
        bins_priv_ref.fetch_add(1, cuda::memory_order_relaxed);
    }
    __syncthreads();
    for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
        if (bins_priv[b] > 0) {
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bins_ref(bins[b]);
            bins_ref.fetch_add(bins_priv[b], cuda::memory_order_relaxed);
        }
    }
}

/* Fig 9.10 */
__global__ void histogram_kernel_10(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {
    __shared__ unsigned int bins_s[NUM_BINS];
    for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
        bins_s[b] = 0;
    }
    __syncthreads();

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width * height) {
        unsigned char b = image[i];
        cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bins_s_ref(bins_s[b]);
        bins_s_ref.fetch_add(1, cuda::memory_order_relaxed);
    }
    __syncthreads();
    for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
        if (bins_s[b] > 0) {
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bins_ref(bins[b]);
            bins_ref.fetch_add(bins_s[b], cuda::memory_order_relaxed);
        }
    }
}

/* Fig 9.12 */
__global__ void histogram_kernel_12(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {
    __shared__ unsigned int bins_s[NUM_BINS];
    for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
        bins_s[b] = 0;
    }
    __syncthreads();

    unsigned int segment = COARSE_FACTOR * blockIdx.x * blockDim.x;
    for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        unsigned int i = segment + threadIdx.x * COARSE_FACTOR + c;
        if (i < width * height) {
            unsigned char b = image[i];
            cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bins_s_ref(bins_s[b]);
            bins_s_ref.fetch_add(1, cuda::memory_order_relaxed);
        }
    }
    __syncthreads();
    for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
        if (bins_s[b] > 0) {
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bins_ref(bins[b]);
            bins_ref.fetch_add(bins_s[b], cuda::memory_order_relaxed);
        }
    }
}

/* Fig 9.14 */
__global__ void histogram_kernel_14(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {
    __shared__ unsigned int bins_s[NUM_BINS];
    for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
        bins_s[b] = 0;
    }
    __syncthreads();
    unsigned int segment = COARSE_FACTOR * blockIdx.x * blockDim.x;
    for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        unsigned int i = segment + c * blockDim.x + threadIdx.x;
        if (i < width * height) {
            unsigned char b = image[i];
            cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bins_s_ref(bins_s[b]);
            bins_s_ref.fetch_add(1, cuda::memory_order_relaxed);
        }
    }
    __syncthreads();
    for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
        if (bins_s[b] > 0) {
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bins_ref(bins[b]);
            bins_ref.fetch_add(bins_s[b], cuda::memory_order_relaxed);
        }
    }
}

/* Fig 9.15 */
__global__ void histogram_kernel_15(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {
    __shared__ unsigned int bins_s[NUM_BINS];
    for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
        bins_s[b] = 0;
    }
    __syncthreads();
    unsigned int segment = COARSE_FACTOR * blockIdx.x * blockDim.x;
    if (segment + threadIdx.x < width * height) {
        unsigned int b = image[segment + threadIdx.x];
        unsigned int bin_r = 1;
        for (unsigned int c = 1; c < COARSE_FACTOR; ++c) {
            unsigned int i = segment + c * blockDim.x + threadIdx.x;
            if (i < width * height) {
                unsigned char bNext = image[i];
                if (bNext == b) {
                    ++bin_r;
                } else {
                    cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bins_s_ref(bins_s[b]);
                    bins_s_ref.fetch_add(bin_r, cuda::memory_order_relaxed);
                    b = bNext;
                    bin_r = 1;
                }
            }
        }
        cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bins_s_ref(bins_s[b]);
        bins_s_ref.fetch_add(bin_r, cuda::memory_order_relaxed);
    }
    __syncthreads();
    for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
        if (bins_s[b] > 0) {
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bins_ref(bins[b]);
            bins_ref.fetch_add(bins_s[b], cuda::memory_order_relaxed);
        }
    }
}

int main(int argc, const char* argv[]) {
    if (argc < 5) {
        printf("usage: %s filename width height mode [iterations]\n", argv[0]);
        printf("\tmode is 0=cpu, 1-6 for different flavors of GPU\n");
        exit(1);
    }

    int width = atoi(argv[2]);
    int height = atoi(argv[3]);
    int mode = atoi(argv[4]);
    int iters = argc > 5 ? atoi(argv[5]) : 1;

    if (width < 2 || height < 2 || mode < Modes::CPU || mode > Modes::GPU_SIMPLE) {
        printf("Please enter reasonable values\n");
        exit(1);
    }

    FILE* f = fopen(argv[1], "r");
    if (f == NULL) {
        printf("Unable to open file %s", argv[1]);
        return 1;
    }

    unsigned int img_size = height * width;
    unsigned int bin_data_size = NUM_BINS * sizeof(int);

    unsigned char* data_host = (unsigned char*)malloc(img_size);
    unsigned int* bins_host = (unsigned int*)malloc(bin_data_size);

    fread(data_host, img_size, 1, f);
    fclose(f);

    for (int i = 0; i < NUM_BINS; i++) {
        bins_host[i] = 0;
    }

    unsigned char* data_device = 0;
    unsigned int* bins_device = 0;

    if (mode != Modes::CPU) {
        CUDACHECK(cudaMalloc((void**)&data_device, img_size));
        CUDACHECK(cudaMalloc((void**)&bins_device, bin_data_size));
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int iter = 0; iter < iters; iter++) {
        if (mode == Modes::CPU) {
            histogram_sequential(data_host, bins_host, width, height);
        } else {
            CUDACHECK(cudaMemcpy(data_device, data_host, img_size, cudaMemcpyHostToDevice));
            CUDACHECK(cudaMemcpy(bins_device, bins_host, bin_data_size, cudaMemcpyHostToDevice));

            dim3 dimGrid(ceil(width * height / 1024.0));
            if (mode == Modes::GPU_SIMPLE) {
                histogram_kernel_6<<<dimGrid, dim3(1024)>>>(data_device, bins_device, width, height);
            } else {
                // 6, 9, 10, 12, 14, 15
                printf("Implement me\n");
            }

            CUDACHECK(cudaGetLastError());
            CUDACHECK(cudaDeviceSynchronize()); // the next call is blocking so does this have any effect?
            CUDACHECK(cudaMemcpy(bins_host, bins_device, bin_data_size, cudaMemcpyDeviceToHost));
        }
    }
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    if (mode != Modes::CPU) {
        printf("Timing %f ms\n", milliseconds);
    }

    print_bins(bins_host);

    if (mode != Modes::CPU) {
        printf("Cleaning up device resources\n");
        CUDACHECK(cudaFree(data_device));
        CUDACHECK(cudaFree(bins_device));
    }

    free(data_host);
    free(bins_host);

    return 0;
}
