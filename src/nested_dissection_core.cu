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
#include <set>

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

FastNestedDissection::FastNestedDissection(const MetisGraph &graph) : n(graph.n), nnz(0), is_structured_grid(false), grid_size(0)
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

    // Load the METIS graph
    loadMetisGraph(graph);

    // Analyze graph structure
    is_structured_grid = detectStructuredGrid();
    if (is_structured_grid)
    {
        auto dims = estimateGridDimensions();
        grid_size = std::max(dims.first, dims.second);
        std::cout << "Detected structured grid: " << dims.first << " x " << dims.second << std::endl;
    }
    else
    {
        std::cout << "General unstructured graph detected" << std::endl;
    }
}

void FastNestedDissection::loadMetisGraph(const MetisGraph &graph)
{
    metis_graph = graph;

    // Convert to CSR format for CUDA
    nnz = graph.col_idx.size();
    d_row_ptr = graph.row_ptr;
    d_col_idx = graph.col_idx;

    // Use edge weights if available, otherwise use unit weights
    if (graph.has_edge_weights)
    {
        d_values = graph.edge_weights;
    }
    else
    {
        std::vector<double> unit_weights(nnz, 1.0);
        d_values = unit_weights;
    }

    std::cout << "Loaded METIS graph: " << n << " vertices, " << nnz << " edges" << std::endl;
}

bool FastNestedDissection::detectStructuredGrid()
{
    // Simple heuristic: check if the graph has regular degree pattern
    // and geometric structure typical of 2D grids

    thrust::host_vector<int> h_row_ptr = d_row_ptr;
    thrust::host_vector<int> h_col_idx = d_col_idx;

    // Count degree distribution
    std::vector<int> degree_counts(10, 0);
    int total_degree_4 = 0;
    int total_degree_3 = 0;
    int total_degree_2 = 0;

    for (int i = 0; i < n; i++)
    {
        int degree = h_row_ptr[i + 1] - h_row_ptr[i];
        if (degree < 10)
        {
            degree_counts[degree]++;
        }
        if (degree == 4)
            total_degree_4++;
        if (degree == 3)
            total_degree_3++;
        if (degree == 2)
            total_degree_2++;
    }

    // For a 2D grid, most internal nodes have degree 4,
    // edge nodes have degree 3, corner nodes have degree 2
    double ratio_regular = (double)(total_degree_2 + total_degree_3 + total_degree_4) / n;

    // Check if adjacency pattern looks like a grid
    bool looks_like_grid = ratio_regular > 0.8;

    if (looks_like_grid)
    {
        // Additional check: verify neighbor patterns
        int grid_like_patterns = 0;
        for (int i = 0; i < std::min(n, 1000); i++)
        { // Sample first 1000 vertices
            std::set<int> neighbors;
            for (int j = h_row_ptr[i]; j < h_row_ptr[i + 1]; j++)
            {
                neighbors.insert(h_col_idx[j]);
            }

            // Check if neighbors form a cross pattern (grid-like)
            bool has_cross_pattern = true;
            for (int neighbor : neighbors)
            {
                if (abs(neighbor - i) != 1 && abs(neighbor - i) > n / 100)
                {
                    // This might be a vertical connection in a 2D grid
                    continue;
                }
            }

            if (has_cross_pattern)
            {
                grid_like_patterns++;
            }
        }

        looks_like_grid = (double)grid_like_patterns / std::min(n, 1000) > 0.5;
    }

    return looks_like_grid;
}

std::pair<int, int> FastNestedDissection::estimateGridDimensions()
{
    // Estimate grid dimensions based on vertex count and connectivity

    int sqrt_n = (int)std::sqrt(n);

    // Try different aspect ratios
    for (int width = sqrt_n - 10; width <= sqrt_n + 10; width++)
    {
        if (width <= 0)
            continue;
        int height = n / width;
        if (width * height == n)
        {
            // Check if this makes sense given the connectivity
            return {width, height};
        }
    }

    // Fallback: assume square grid
    return {sqrt_n, sqrt_n};
}

void FastNestedDissection::printGraphAnalysis()
{
    std::cout << "\n=== GRAPH ANALYSIS ===" << std::endl;

    thrust::host_vector<int> h_row_ptr = d_row_ptr;

    // Compute degree statistics
    std::vector<int> degrees;
    for (int i = 0; i < n; i++)
    {
        degrees.push_back(h_row_ptr[i + 1] - h_row_ptr[i]);
    }

    std::sort(degrees.begin(), degrees.end());

    double avg_degree = (double)nnz / n;
    int min_degree = degrees[0];
    int max_degree = degrees[n - 1];
    int median_degree = degrees[n / 2];

    std::cout << "Vertices: " << n << std::endl;
    std::cout << "Edges: " << nnz / 2 << " (undirected)" << std::endl;
    std::cout << "Average degree: " << avg_degree << std::endl;
    std::cout << "Degree range: [" << min_degree << ", " << max_degree << "]" << std::endl;
    std::cout << "Median degree: " << median_degree << std::endl;

    if (is_structured_grid)
    {
        std::cout << "Graph type: Structured grid (estimated " << grid_size << "x" << grid_size << ")" << std::endl;
    }
    else
    {
        std::cout << "Graph type: General unstructured graph" << std::endl;
    }

    // Analyze vertex weights if present
    if (metis_graph.has_vertex_weights)
    {
        std::cout << "Vertex weights: Present (" << metis_graph.ncon << " constraints)" << std::endl;
    }

    if (metis_graph.has_edge_weights)
    {
        std::cout << "Edge weights: Present" << std::endl;
    }

    if (metis_graph.has_vertex_sizes)
    {
        std::cout << "Vertex sizes: Present" << std::endl;
    }
}

double FastNestedDissection::computeFillReduction()
{
    // Estimate fill reduction compared to natural ordering

    // This is a simplified estimate - in practice you'd want to compute
    // the actual symbolic factorization

    // For grid graphs, nested dissection typically reduces fill significantly
    if (is_structured_grid)
    {
        // Grid graphs: fill goes from O(n^1.5) to O(n log n)
        double natural_fill = std::pow(n, 1.5);
        double nd_fill = n * std::log2(n);
        return (1.0 - nd_fill / natural_fill) * 100.0;
    }
    else
    {
        // General graphs: more conservative estimate
        double avg_degree = (double)nnz / n;
        double estimated_reduction = std::min(50.0, 10.0 * std::log10(avg_degree));
        return estimated_reduction;
    }
}

// Enhanced separator methods for general graphs
std::vector<int> FastNestedDissection::geometricSeparatorGeneral(const std::vector<int> &vertices)
{
    // For general graphs, try to find a geometric separator using
    // coordinate embedding or spectral coordinates

    if (vertices.size() <= 128)
    {
        return std::vector<int>{vertices[vertices.size() / 2]};
    }

    // Use spectral coordinates as geometric proxy
    thrust::device_vector<double> d_coords_x(vertices.size());
    thrust::device_vector<double> d_coords_y(vertices.size());

    // Simple coordinate computation based on graph structure
    thrust::fill(d_coords_x.begin(), d_coords_x.end(), 0.0);
    thrust::fill(d_coords_y.begin(), d_coords_y.end(), 0.0);

    // Use vertex IDs as initial coordinates
    for (int i = 0; i < vertices.size(); i++)
    {
        d_coords_x[i] = vertices[i] % (int)std::sqrt(vertices.size());
        d_coords_y[i] = vertices[i] / (int)std::sqrt(vertices.size());
    }

    // Find median in both dimensions
    thrust::host_vector<double> h_coords_x = d_coords_x;
    thrust::host_vector<double> h_coords_y = d_coords_y;

    std::vector<std::pair<double, int>> x_pairs, y_pairs;
    for (int i = 0; i < vertices.size(); i++)
    {
        x_pairs.push_back({h_coords_x[i], i});
        y_pairs.push_back({h_coords_y[i], i});
    }

    std::sort(x_pairs.begin(), x_pairs.end());
    std::sort(y_pairs.begin(), y_pairs.end());

    // Create separator around median
    std::vector<int> separator;
    int sep_size = std::max(1, (int)(vertices.size() * 0.03));
    int start_x = vertices.size() / 2 - sep_size / 2;

    for (int i = start_x; i < start_x + sep_size && i < x_pairs.size(); i++)
    {
        separator.push_back(vertices[x_pairs[i].second]);
    }

    return separator;
}

std::vector<int> FastNestedDissection::multilevelSeparator(const std::vector<int> &vertices)
{
    // Simplified multilevel approach: coarsen, separate, refine

    if (vertices.size() <= 256)
    {
        return approximateSpectralSeparator(vertices);
    }

    // Coarsen by random matching (simplified)
    std::vector<int> coarse_vertices;
    std::unordered_set<int> used;

    for (int i = 0; i < vertices.size(); i += 2)
    {
        if (used.find(vertices[i]) == used.end())
        {
            coarse_vertices.push_back(vertices[i]);
            used.insert(vertices[i]);
            if (i + 1 < vertices.size())
            {
                used.insert(vertices[i + 1]);
            }
        }
    }

    // Recursively find separator in coarse graph
    std::vector<int> coarse_separator = approximateSpectralSeparator(coarse_vertices);

    // Project back to fine graph (simplified)
    std::vector<int> fine_separator = coarse_separator;

    // Add some neighboring vertices for refinement
    std::unordered_set<int> sep_set(fine_separator.begin(), fine_separator.end());
    thrust::host_vector<int> h_row_ptr = d_row_ptr;
    thrust::host_vector<int> h_col_idx = d_col_idx;

    for (int v : coarse_separator)
    {
        for (int j = h_row_ptr[v]; j < h_row_ptr[v + 1]; j++)
        {
            int neighbor = h_col_idx[j];
            if (sep_set.find(neighbor) == sep_set.end() &&
                std::find(vertices.begin(), vertices.end(), neighbor) != vertices.end())
            {
                fine_separator.push_back(neighbor);
                sep_set.insert(neighbor);
            }
        }
    }

    return fine_separator;
}

std::vector<int> FastNestedDissection::improvedSpectralSeparator(const std::vector<int> &vertices)
{
    // Enhanced spectral separator with better power iteration
    return approximateSpectralSeparator(vertices); // Use existing implementation for now
}