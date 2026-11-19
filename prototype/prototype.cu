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

#include "json.hpp"
#include <chrono>
#include <cstdio>
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <format>
#include <fstream>
#include <iostream>
#include <stdio.h>

using nlohmann::json;
using std::string;

enum Modes { CPU = 0, GPU_SIMPLE };
static const int NUM_BINS = 256;
static const int COARSE_FACTOR = 3;
static string JSON_LOG = "log.json";

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

class Timer {
  private:
    // Type aliases to make accessing nested type easier
    using Clock = std::chrono::steady_clock;
    using Second = std::chrono::duration<double, std::ratio<1>>;

    std::chrono::time_point<Clock> m_beg{Clock::now()};
    bool isCpu;
    cudaEvent_t start, stop;

  public:
    Timer(bool isCpu) {
        this->isCpu = isCpu;

        if (!isCpu) {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
        }
    }

    void reset() { m_beg = Clock::now(); }

    double elapsedMs() const {
        if (isCpu) {
            return std::chrono::duration_cast<Second>(Clock::now() - m_beg).count() * 1000;
        } else {
            cudaEventRecord(stop);

            cudaEventSynchronize(stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);

            return milliseconds;
        }
    }
};

void print_bins(unsigned int* bins) {
    for (int i = 0; i < NUM_BINS; i++) {
        if (bins[i] != 0) {
            printf("\"%d\": %d%s", i, bins[i], i == (NUM_BINS - 1) ? "" : ",");
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

    string fname = argv[1];
    int width = atoi(argv[2]);
    int height = atoi(argv[3]);
    int mode = atoi(argv[4]);
    int iters = argc > 5 ? atoi(argv[5]) : 1;

    if (width < 2 || height < 2 || mode < Modes::CPU) {
        printf("Please enter reasonable values\n");
        exit(1);
    }

    FILE* f = fopen(fname.c_str(), "r");
    if (f == NULL) {
        printf("Unable to open file %s\n", fname.c_str());
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

    Timer t(mode == Modes::CPU);

    for (int iter = 0; iter < iters; iter++) {
        if (mode == Modes::CPU) {
            histogram_sequential(data_host, bins_host, width, height);
        } else {
            CUDACHECK(cudaMemcpy(data_device, data_host, img_size, cudaMemcpyHostToDevice));
            CUDACHECK(cudaMemcpy(bins_device, bins_host, bin_data_size, cudaMemcpyHostToDevice));

            dim3 dimGrid(ceil(width * height / 1024.0));
            if (mode == Modes::GPU_SIMPLE) {
                histogram_kernel_6<<<dimGrid, dim3(1024)>>>(data_device, bins_device, width, height);
            } else if (mode == 15) {
                histogram_kernel_15<<<dimGrid, dim3(1024)>>>(data_device, bins_device, width, height);
            } else {
                // the rest
                printf("Implement me\n");
            }
            CUDACHECK(cudaGetLastError());
            CUDACHECK(cudaDeviceSynchronize()); // the next call is blocking so does this have any effect?
            CUDACHECK(cudaMemcpy(bins_host, bins_device, bin_data_size, cudaMemcpyDeviceToHost));
        }
    }

    printf("Time for size %d, algorithm %d, repeated %d: %f ms\n", height * width, mode, iters, t.elapsedMs());

    // print_bins(bins_host);

    json j;
    j["timeMs"] = t.elapsedMs();

    std::vector<int> v;
    for (int i = 0; i < NUM_BINS; i++) {
        v.push_back(bins_host[i]);
    }
    j["results"] = v;

    if (mode != Modes::CPU) {
        // printf("Cleaning up device resources\n");
        CUDACHECK(cudaFree(data_device));
        CUDACHECK(cudaFree(bins_device));
    }

    free(data_host);
    free(bins_host);

    std::ifstream infile(JSON_LOG);
    json data = json::parse(infile);
    infile.close();

    std::ofstream outfile(JSON_LOG);
    j["meta"]["fname"] = fname;
    j["meta"]["algo"] = mode;
    j["meta"]["iters"] = iters;

    unsigned long milliseconds_since_epoch =
        std::chrono::system_clock::now().time_since_epoch() / std::chrono::milliseconds(1);

    data[std::to_string(milliseconds_since_epoch)] = j;
    outfile << std::setw(4) << data << std::endl;
    outfile.close();

    return 0;
}
