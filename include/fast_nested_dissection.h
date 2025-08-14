#pragma once

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <cusolverSp.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>
#include <memory>
#include <utility>
#include "metis_parser.h"

// Error checking macros
#define CUDA_CHECK(call)                                                                               \
    do                                                                                                 \
    {                                                                                                  \
        cudaError_t err = call;                                                                        \
        if (err != cudaSuccess)                                                                        \
        {                                                                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1);                                                                                   \
        }                                                                                              \
    } while (0)

#define CUSPARSE_CHECK(call)                                                           \
    do                                                                                 \
    {                                                                                  \
        cusparseStatus_t err = call;                                                   \
        if (err != CUSPARSE_STATUS_SUCCESS)                                            \
        {                                                                              \
            fprintf(stderr, "cuSPARSE error at %s:%d: %d\n", __FILE__, __LINE__, err); \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

// Forward declarations
class GPUMemoryPool;
struct SeparatorNode;
class QualityMetrics;

// GPU Memory Pool for optimization
class GPUMemoryPool
{
private:
    std::vector<void *> free_blocks;
    std::vector<size_t> block_sizes;
    size_t total_allocated = 0;

public:
    void *allocate(size_t size);
    void deallocate(void *ptr, size_t size);
    ~GPUMemoryPool();
};

extern GPUMemoryPool memory_pool;

// Separator tree node structure
struct SeparatorNode
{
    std::vector<int> vertices;
    std::vector<int> A_vertices;
    std::vector<int> B_vertices;
    std::unique_ptr<SeparatorNode> left;
    std::unique_ptr<SeparatorNode> right;
    int level;

    SeparatorNode(int lvl = 0);
};

// Main nested dissection class - now supports METIS graphs
class FastNestedDissection
{
private:
    cudaStream_t computation_stream = 0;
    cusparseHandle_t cusparse_handle;
    cublasHandle_t cublas_handle;

    int n;
    int nnz;
    bool is_structured_grid; // True for grid graphs, false for general graphs
    int grid_size;           // Only used for structured grids

    thrust::device_vector<int> d_row_ptr;
    thrust::device_vector<int> d_col_idx;
    thrust::device_vector<double> d_values;

    // Store original METIS graph data
    MetisGraph metis_graph;

    std::unique_ptr<SeparatorNode> separator_tree;
    thrust::device_vector<int> d_perm;

    // Reusable GPU buffers
    thrust::device_vector<double> workspace_vec1;
    thrust::device_vector<double> workspace_vec2;
    thrust::device_vector<int> workspace_int;

    // Private methods
    void buildNestedDissectionTreeFast();
    std::pair<std::vector<int>, std::vector<int>> fastConnectedComponents(
        const std::vector<int> &remaining, const std::vector<int> &separator);
    void generatePermutationRecursive(SeparatorNode *node, std::vector<int> &permutation);
    void printTreeRecursive(SeparatorNode *node, std::string indent);

    // Graph analysis methods
    bool detectStructuredGrid();
    std::pair<int, int> estimateGridDimensions();

    // Enhanced separator methods for general graphs
    std::vector<int> geometricSeparatorGeneral(const std::vector<int> &vertices);
    std::vector<int> multilevelSeparator(const std::vector<int> &vertices);
    std::vector<int> improvedSpectralSeparator(const std::vector<int> &vertices);

public:
    // Separator methods that need device lambda access
    std::vector<int> fastGeometricSeparator(const std::vector<int> &vertices,
                                            int start_x = 0, int start_y = 0,
                                            int width = 0, int height = 0);
    std::vector<int> approximateSpectralSeparator(const std::vector<int> &vertices);
    void simpleRandomWalkStep(thrust::device_vector<double> &d_x,
                              thrust::device_vector<double> &d_y,
                              const std::vector<int> &vertices);
    std::vector<int> partitionByEigenvector(const std::vector<int> &vertices,
                                            const thrust::device_vector<double> &eigenvec);

    void createSubgraphMapping(const std::vector<int> &vertices,
                               thrust::device_vector<int> &d_vertex_map,
                               thrust::device_vector<int> &d_reverse_map);
    void performSpMVSubgraph(const thrust::device_vector<double> &d_x,
                             thrust::device_vector<double> &d_y,
                             const std::vector<int> &vertices);

    std::pair<std::vector<int>, std::vector<int>> gpuConnectedComponents(
        const std::vector<int> &remaining, const std::vector<int> &separator);

public:
    // Constructors
    FastNestedDissection(int matrix_size, int grid_sz = 0); // Original constructor
    FastNestedDissection(const MetisGraph &graph);          // New METIS constructor
    ~FastNestedDissection();

    // Matrix loading methods
    void loadMatrix(const std::vector<int> &row_ptr,
                    const std::vector<int> &col_idx,
                    const std::vector<double> &values);
    void loadMetisGraph(const MetisGraph &graph);

    // Main algorithm methods
    void performNestedDissection();
    void generatePermutation();
    std::vector<int> getPermutation();
    void printTreeInfo();

    // Analysis methods
    void printGraphAnalysis();
    double computeFillReduction();
};

// Enhanced quality improvement metrics for general graphs
class QualityMetrics
{
public:
    static double computeSeparatorQuality(const std::vector<int> &separator,
                                          const std::vector<int> &A_vertices,
                                          const std::vector<int> &B_vertices);
    static std::vector<int> refineSeparator(const std::vector<int> &separator,
                                            const std::vector<int> &A_vertices,
                                            const std::vector<int> &B_vertices,
                                            const std::vector<int> &row_ptr,
                                            const std::vector<int> &col_idx);

    // New methods for general graphs
    static double computeBalanceRatio(const std::vector<int> &A_vertices,
                                      const std::vector<int> &B_vertices);
    static double computeEdgeCut(const std::vector<int> &separator,
                                 const std::vector<int> &A_vertices,
                                 const std::vector<int> &B_vertices,
                                 const std::vector<int> &row_ptr,
                                 const std::vector<int> &col_idx);
    static std::vector<int> kernighanLinRefinement(const std::vector<int> &separator,
                                                   const std::vector<int> &A_vertices,
                                                   const std::vector<int> &B_vertices,
                                                   const std::vector<int> &row_ptr,
                                                   const std::vector<int> &col_idx,
                                                   int max_iterations = 10);
};

// CUDA kernel declarations
__global__ void computeDegreesShared(const int *row_ptr, int n, int *degrees);
__global__ void setupLaplacianFused(const int *row_ptr, const int *col_idx,
                                    double *values, const int *degrees, int n);
__global__ void geometricSeparator2D(int *separator_mask, int grid_size,
                                     int start_x, int start_y, int width, int height);
__global__ void computeVertexCoordinates(const int *row_ptr, const int *col_idx,
                                         int n, double *x_coords, double *y_coords);
__global__ void spectralMatVec(const int *row_ptr, const int *col_idx, const double *values,
                               const double *x, double *y, int n);
__device__ void atomicAddDouble(double *address, double val);

__global__ void sparseMatVec(const int *row_ptr, const int *col_idx,
                             const double *values, const double *x,
                             double *y, int n);
__global__ void sparseMatVecSubgraph(const int *row_ptr, const int *col_idx,
                                     const double *values, const double *x,
                                     double *y, const int *vertex_map,
                                     const int *reverse_map, int sub_n, int n);

__global__ void initializeLabels(int *labels, int n);
__global__ void propagateLabels(const int *row_ptr, const int *col_idx,
                                int *labels, bool *changed,
                                const int *separator_mask, int n);
__global__ void extractComponent(const int *labels, const int *vertices,
                                 int *component_vertices, int *component_size,
                                 int target_label, int num_vertices);