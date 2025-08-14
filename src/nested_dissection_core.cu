#include "../include/fast_nested_dissection.h"
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/remove.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/execution_policy.h>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <chrono>

// Global memory pool instance
GPUMemoryPool memory_pool;

// GPUMemoryPool implementation
void *GPUMemoryPool::allocate(size_t size)
{
    // Find suitable block
    for (int i = 0; i < free_blocks.size(); i++)
    {
        if (block_sizes[i] >= size)
        {
            void *ptr = free_blocks[i];
            free_blocks.erase(free_blocks.begin() + i);
            block_sizes.erase(block_sizes.begin() + i);
            return ptr;
        }
    }
    // Allocate new block
    void *ptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    total_allocated += size;
    return ptr;
}

void GPUMemoryPool::deallocate(void *ptr, size_t size)
{
    free_blocks.push_back(ptr);
    block_sizes.push_back(size);
}

GPUMemoryPool::~GPUMemoryPool()
{
    for (void *ptr : free_blocks)
    {
        cudaFree(ptr);
    }
}

// SeparatorNode implementation
SeparatorNode::SeparatorNode(int lvl) : level(lvl) {}

// CUDA kernels
__global__ void computeDegreesShared(const int *row_ptr, int n, int *degrees)
{
    __shared__ int shared_data[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < n)
    {
        shared_data[tid] = row_ptr[idx + 1] - row_ptr[idx];
        degrees[idx] = shared_data[tid];
    }
}

__global__ void setupLaplacianFused(const int *row_ptr, const int *col_idx,
                                    double *values, const int *degrees, int n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n)
    {
        int degree = degrees[row];
        for (int j = row_ptr[row]; j < row_ptr[row + 1]; j++)
        {
            if (col_idx[j] == row)
            {
                values[j] = degree; // Diagonal
            }
            else
            {
                values[j] = -1.0; // Off-diagonal
            }
        }
    }
}

__global__ void geometricSeparator2D(int *separator_mask, int grid_size,
                                     int start_x, int start_y, int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = width * height;

    if (idx < total_size)
    {
        int local_x = idx % width;
        int local_y = idx / width;
        int global_x = start_x + local_x;
        int global_y = start_y + local_y;
        int global_idx = global_y * grid_size + global_x;

        // Create separator in the middle
        if (local_x == width / 2 || local_y == height / 2)
        {
            separator_mask[global_idx] = 1;
        }
        else
        {
            separator_mask[global_idx] = 0;
        }
    }
}

__device__ void atomicAddDouble(double *address, double val)
{
    unsigned long long int *address_as_ull = (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
}

// FastNestedDissection core implementation
FastNestedDissection::FastNestedDissection(int matrix_size, int grid_sz) : n(matrix_size), nnz(0), grid_size(grid_sz)
{
    CUSPARSE_CHECK(cusparseCreate(&cusparse_handle));
    cublasStatus_t cublas_status = cublasCreate(&cublas_handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "cuBLAS error: %d\n", cublas_status);
        exit(1);
    }

    d_perm.resize(n);
    thrust::sequence(d_perm.begin(), d_perm.end());

    // Pre-allocate workspace
    workspace_vec1.resize(n);
    workspace_vec2.resize(n);
    workspace_int.resize(n);
}

FastNestedDissection::~FastNestedDissection()
{
    cusparseDestroy(cusparse_handle);
    cublasDestroy(cublas_handle);
}

void FastNestedDissection::loadMatrix(const std::vector<int> &row_ptr,
                                      const std::vector<int> &col_idx,
                                      const std::vector<double> &values)
{
    nnz = values.size();
    d_row_ptr = row_ptr;
    d_col_idx = col_idx;
    d_values = values;
}

std::vector<int> FastNestedDissection::fastGeometricSeparator(const std::vector<int> &vertices,
                                                              int start_x, int start_y, int width, int height)
{
    if (grid_size == 0)
    {
        // Fall back to spectral method for unstructured graphs
        return approximateSpectralSeparator(vertices);
    }

    thrust::device_vector<int> d_separator_mask(n, 0);

    int blockSize = 256;
    int gridSize = (width * height + blockSize - 1) / blockSize;

    geometricSeparator2D<<<gridSize, blockSize>>>(
        thrust::raw_pointer_cast(d_separator_mask.data()),
        grid_size, start_x, start_y, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Extract separator vertices
    thrust::host_vector<int> h_mask = d_separator_mask;
    std::vector<int> separator;

    for (int v : vertices)
    {
        if (h_mask[v] == 1)
        {
            separator.push_back(v);
        }
    }

    // Fallback: if no separator found, use simple median split
    if (separator.empty() && vertices.size() > 128)
    {
        int sep_size = std::max(1, (int)(vertices.size() * 0.05));
        int start = vertices.size() / 2 - sep_size / 2;
        for (int i = 0; i < sep_size && start + i < vertices.size(); i++)
        {
            separator.push_back(vertices[start + i]);
        }
    }

    return separator;
}

std::vector<int> FastNestedDissection::approximateSpectralSeparator(const std::vector<int> &vertices)
{
    int sub_n = vertices.size();

    if (sub_n <= 128)
    { // Larger base case
        std::vector<int> separator;
        separator.push_back(vertices[sub_n / 2]);
        return separator;
    }

    // Use workspace vectors
    thrust::device_vector<double> d_x(workspace_vec1.begin(), workspace_vec1.begin() + sub_n);
    thrust::device_vector<double> d_y(workspace_vec2.begin(), workspace_vec2.begin() + sub_n);
    thrust::device_vector<double> d_ones(sub_n, 1.0);

    thrust::fill(d_x.begin(), d_x.end(), 1.0);

    // Only 20 iterations instead of 100
    for (int iter = 0; iter < 20; iter++)
    {
        // Simplified power iteration without full Laplacian construction
        // Just use degree-weighted random walk
        simpleRandomWalkStep(d_x, d_y, vertices);

        // Simple deflation
        double dot_y_ones = thrust::inner_product(d_y.begin(), d_y.end(), d_ones.begin(), 0.0);
        double alpha = -dot_y_ones / sub_n;

        thrust::transform(d_y.begin(), d_y.end(), d_y.begin(),
                          [alpha] __device__(double y)
                          { return y + alpha; });

        // Normalize
        double norm_y = std::sqrt(thrust::inner_product(d_y.begin(), d_y.end(), d_y.begin(), 0.0));
        if (norm_y > 1e-12)
        {
            thrust::transform(d_y.begin(), d_y.end(), d_y.begin(),
                              [norm_y] __device__(double y)
                              { return y / norm_y; });
        }

        d_x = d_y;
    }

    return partitionByEigenvector(vertices, d_x);
}

void FastNestedDissection::simpleRandomWalkStep(thrust::device_vector<double> &d_x,
                                                thrust::device_vector<double> &d_y,
                                                const std::vector<int> &vertices)
{
    // This is a simplified placeholder - in practice you'd implement
    // a fast sparse matrix-vector multiply here
    thrust::copy(d_x.begin(), d_x.end(), d_y.begin());
}

std::vector<int> FastNestedDissection::partitionByEigenvector(const std::vector<int> &vertices,
                                                              const thrust::device_vector<double> &eigenvec)
{
    thrust::host_vector<double> h_eigenvec = eigenvec;
    std::vector<std::pair<double, int>> eigen_pairs;

    for (int i = 0; i < vertices.size(); i++)
    {
        eigen_pairs.push_back({h_eigenvec[i], i});
    }

    std::sort(eigen_pairs.begin(), eigen_pairs.end());

    // Smaller separator size
    int median_idx = eigen_pairs.size() / 2;
    std::vector<int> separator;

    int sep_size = std::max(1, (int)(vertices.size() * 0.05)); // Smaller separator
    int start = std::max(0, median_idx - sep_size / 2);
    int end = std::min((int)eigen_pairs.size(), start + sep_size);

    for (int i = start; i < end; i++)
    {
        separator.push_back(vertices[eigen_pairs[i].second]);
    }

    return separator;
}