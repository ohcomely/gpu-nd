#include "../include/nested_dissection.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <algorithm>
#include <random>
#include <cassert>
#include <numeric>
#include <chrono>
#include <map>
#include <set>
#include <cmath>
#include <climits>
#include <curand.h>
#include <curand_kernel.h>

// Enhanced GPU kernels for heavy edge matching and graph coarsening
// Add these to your nested_dissection.cu file

// Additional kernel for counting coarse vertex degrees
__global__ void count_coarse_vertex_degrees(int *row_ptr, int *col_idx, int *vertex_map,
                                            int *degree_count, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        int coarse_v = vertex_map[idx];

        for (int e = row_ptr[idx]; e < row_ptr[idx + 1]; e++)
        {
            int neighbor = col_idx[e];
            int coarse_neighbor = vertex_map[neighbor];

            if (coarse_v != coarse_neighbor)
            {
                atomicAdd(&degree_count[coarse_v], 1);
            }
        }
    }
}

__global__ void initialize_matching_arrays(int *match, int *vertex_weights, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        match[idx] = -1;
        vertex_weights[idx] = 1;
    }
}

__global__ void heavy_edge_matching_phase1(int *row_ptr, int *col_idx, int *edge_weights,
                                           int *match, int *proposals, curandState *states, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && match[idx] == -1)
    {
        int best_neighbor = -1;
        int max_weight = -1;

        // Find heaviest unmatched neighbor
        for (int e = row_ptr[idx]; e < row_ptr[idx + 1]; e++)
        {
            int neighbor = col_idx[e];
            int weight = edge_weights[e];

            if (neighbor != idx && match[neighbor] == -1 && weight > max_weight)
            {
                max_weight = weight;
                best_neighbor = neighbor;
            }
        }

        proposals[idx] = best_neighbor;
    }
    else
    {
        proposals[idx] = -1;
    }
}

__global__ void heavy_edge_matching_phase2(int *proposals, int *match, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        int proposed_neighbor = proposals[idx];
        if (proposed_neighbor != -1 && proposals[proposed_neighbor] == idx)
        {
            // Mutual proposal - create match
            if (idx < proposed_neighbor)
            {
                match[idx] = proposed_neighbor;
                match[proposed_neighbor] = idx;
            }
        }
    }
}

__global__ void compute_coarse_vertex_mapping(int *match, int *vertex_map, int *coarse_vertex_count, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        if (match[idx] == -1)
        {
            // Unmatched vertex gets its own new vertex ID
            vertex_map[idx] = atomicAdd(coarse_vertex_count, 1);
        }
        else if (idx < match[idx])
        {
            // For matched pair, smaller ID gets the new vertex ID
            int new_id = atomicAdd(coarse_vertex_count, 1);
            vertex_map[idx] = new_id;
            vertex_map[match[idx]] = new_id;
        }
    }
}

__global__ void count_coarse_edges(int *row_ptr, int *col_idx, int *vertex_map,
                                   int *coarse_edge_count, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        int coarse_v = vertex_map[idx];

        for (int e = row_ptr[idx]; e < row_ptr[idx + 1]; e++)
        {
            int neighbor = col_idx[e];
            int coarse_neighbor = vertex_map[neighbor];

            if (coarse_v != coarse_neighbor)
            {
                atomicAdd(coarse_edge_count, 1);
            }
        }
    }
}

__global__ void build_coarse_graph_structure(int *old_row_ptr, int *old_col_idx, int *old_edge_weights,
                                             int *vertex_map, int old_n, int coarse_n,
                                             int *new_row_ptr, int *edge_positions, int *temp_edges,
                                             int *temp_weights, int *temp_neighbors)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < old_n)
    {
        int coarse_v = vertex_map[idx];

        for (int e = old_row_ptr[idx]; e < old_row_ptr[idx + 1]; e++)
        {
            int neighbor = old_col_idx[e];
            int coarse_neighbor = vertex_map[neighbor];
            int weight = old_edge_weights[e];

            if (coarse_v != coarse_neighbor)
            {
                int pos = atomicAdd(&edge_positions[coarse_v], 1);
                int global_pos = new_row_ptr[coarse_v] + pos;

                temp_neighbors[global_pos] = coarse_neighbor;
                temp_weights[global_pos] = weight;
            }
        }
    }
}

__global__ void merge_parallel_edges(int *row_ptr, int *col_idx, int *edge_weights,
                                     int *new_col_idx, int *new_edge_weights, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        int start = row_ptr[idx];
        int end = row_ptr[idx + 1];
        int degree = end - start;

        if (degree > 0)
        {
            // Sort edges by neighbor ID (simple bubble sort for small degrees)
            for (int i = 0; i < degree - 1; i++)
            {
                for (int j = start; j < start + degree - 1 - i; j++)
                {
                    if (col_idx[j] > col_idx[j + 1])
                    {
                        // Swap neighbors
                        int temp_neighbor = col_idx[j];
                        col_idx[j] = col_idx[j + 1];
                        col_idx[j + 1] = temp_neighbor;

                        // Swap weights
                        int temp_weight = edge_weights[j];
                        edge_weights[j] = edge_weights[j + 1];
                        edge_weights[j + 1] = temp_weight;
                    }
                }
            }

            // Merge parallel edges
            int write_pos = start;
            for (int read_pos = start; read_pos < end; read_pos++)
            {
                if (write_pos == start || col_idx[read_pos] != col_idx[write_pos - 1])
                {
                    // New unique neighbor
                    col_idx[write_pos] = col_idx[read_pos];
                    edge_weights[write_pos] = edge_weights[read_pos];
                    write_pos++;
                }
                else
                {
                    // Parallel edge - add weight to previous
                    edge_weights[write_pos - 1] += edge_weights[read_pos];
                }
            }

            // Update row pointer for next vertex
            if (idx < n - 1)
            {
                row_ptr[idx + 1] = write_pos;
            }
        }
    }
}

__global__ void compute_vertex_weights_coarse(int *match, int *old_vertex_weights,
                                              int *new_vertex_weights, int *vertex_map, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        int coarse_v = vertex_map[idx];
        atomicAdd(&new_vertex_weights[coarse_v], old_vertex_weights[idx]);
    }
}

// Utility kernel for parallel prefix sum (simplified version)
__global__ void parallel_prefix_sum(int *input, int *output, int n)
{
    extern __shared__ int temp[];
    int tid = threadIdx.x;
    int offset = 1;

    // Load input into shared memory
    if (blockIdx.x * blockDim.x + tid < n)
    {
        temp[2 * tid] = input[2 * blockIdx.x * blockDim.x + 2 * tid];
        temp[2 * tid + 1] = input[2 * blockIdx.x * blockDim.x + 2 * tid + 1];
    }
    else
    {
        temp[2 * tid] = 0;
        temp[2 * tid + 1] = 0;
    }

    // Build sum in place up the tree
    for (int d = blockDim.x; d > 0; d >>= 1)
    {
        __syncthreads();
        if (tid < d)
        {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    // Clear the last element
    if (tid == 0)
        temp[2 * blockDim.x - 1] = 0;

    // Traverse down tree & build scan
    for (int d = 1; d < 2 * blockDim.x; d *= 2)
    {
        offset >>= 1;
        __syncthreads();
        if (tid < d)
        {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    // Write results to device memory
    if (2 * blockIdx.x * blockDim.x + 2 * tid < n)
    {
        output[2 * blockIdx.x * blockDim.x + 2 * tid] = temp[2 * tid];
    }
    if (2 * blockIdx.x * blockDim.x + 2 * tid + 1 < n)
    {
        output[2 * blockIdx.x * blockDim.x + 2 * tid + 1] = temp[2 * tid + 1];
    }
}

template <typename RandomIt>
void simple_random_shuffle(RandomIt first, RandomIt last)
{
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(first, last, g);
}

// CUDA kernel implementations
__global__ void init_random_states(curandState *states, int n, unsigned long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void compute_vertex_degrees(int *row_ptr, int n, int *degrees)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        degrees[idx] = row_ptr[idx + 1] - row_ptr[idx];
    }
}

__global__ void heavy_edge_matching_kernel(int *row_ptr, int *col_idx, int *edge_weights,
                                           int *match, curandState *states, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && match[idx] == -1)
    {
        int best_neighbor = -1;
        int max_weight = -1;

        // Find heaviest edge
        for (int e = row_ptr[idx]; e < row_ptr[idx + 1]; e++)
        {
            int neighbor = col_idx[e];
            if (neighbor != idx && match[neighbor] == -1 && edge_weights[e] > max_weight)
            {
                max_weight = edge_weights[e];
                best_neighbor = neighbor;
            }
        }

        // Try to match with best neighbor
        if (best_neighbor != -1)
        {
            int old = atomicCAS(&match[best_neighbor], -1, idx);
            if (old == -1)
            {
                match[idx] = best_neighbor;
            }
        }
    }
}

__global__ void contract_graph_kernel(int *old_row_ptr, int *old_col_idx, int *old_edge_weights,
                                      int *match, int *vertex_map, int old_n,
                                      int *new_row_ptr, int *new_col_idx, int *new_edge_weights,
                                      int *new_vertex_weights, int new_n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < old_n)
    {
        // Simplified contraction - in practice this would be more complex
        int new_vertex = vertex_map[idx];
        if (new_vertex < new_n)
        {
            atomicAdd(&new_vertex_weights[new_vertex], 1);
        }
    }
}

// Constructor
ImprovedGPUNestedDissection::ImprovedGPUNestedDissection(int max_v) : max_vertices(max_v)
{
    // Allocate device memory
    cudaMalloc(&d_partition, max_vertices * sizeof(int));
    cudaMalloc(&d_degrees, max_vertices * sizeof(int));
    cudaMalloc(&d_match, max_vertices * sizeof(int));
    cudaMalloc(&d_vertex_map, max_vertices * sizeof(int));
    cudaMalloc(&d_temp_array, max_vertices * sizeof(int));
    cudaMalloc(&d_rand_states, max_vertices * sizeof(curandState));

    // Initialize random states
    dim3 grid((max_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    init_random_states<<<grid, block>>>(d_rand_states, max_vertices,
                                        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    cudaDeviceSynchronize();
}

// Destructor
ImprovedGPUNestedDissection::~ImprovedGPUNestedDissection()
{
    cudaFree(d_partition);
    cudaFree(d_degrees);
    cudaFree(d_match);
    cudaFree(d_vertex_map);
    cudaFree(d_temp_array);
    cudaFree(d_rand_states);
}

// Main compute ordering method
// Improved Compute Ordering Function
// Replace the existing compute_ordering function in nested_dissection.cu

std::vector<int> ImprovedGPUNestedDissection::compute_ordering(const Graph &graph)
{
    h_ordering.clear();
    h_ordering.reserve(graph.n_vertices);

    std::vector<int> separator_stack;
    std::vector<int> all_vertices(graph.n_vertices);
    std::iota(all_vertices.begin(), all_vertices.end(), 0);

    std::queue<DissectionTask> task_queue;
    task_queue.emplace(all_vertices, 0);

    int max_levels = static_cast<int>(std::log2(graph.n_vertices)) + 2;

    while (!task_queue.empty())
    {
        DissectionTask task = task_queue.front();
        task_queue.pop();

        std::cout << "Processing task with " << task.vertices.size()
                  << " vertices at level " << task.level << std::endl;

        if (task.level > max_levels || task.vertices.size() <= MIN_SUBGRAPH_SIZE)
        {
            // Base case: add vertices to ordering
            for (int v : task.vertices)
            {
                h_ordering.push_back(v);
            }
            continue;
        }

        // Use multilevel approach for larger subgraphs (lowered threshold)
        if (task.vertices.size() > 200)
        { // Reduced from 1000 to 200
            std::cout << "  Attempting multilevel partition..." << std::endl;
            Partition partition = multilevel_partition(graph, task.vertices);
            partition.print_stats();

            // Verify partition quality
            int total_vertices = partition.left_vertices.size() +
                                 partition.right_vertices.size() +
                                 partition.separator_vertices.size();

            if (total_vertices == task.vertices.size())
            {
                // Partition is valid, use it
                std::cout << "  Multilevel partition successful" << std::endl;

                // Store separators for later
                for (int v : partition.separator_vertices)
                {
                    separator_stack.push_back(v);
                }

                // Add subtasks
                if (!partition.left_vertices.empty())
                {
                    task_queue.emplace(partition.left_vertices, task.level + 1);
                }
                if (!partition.right_vertices.empty())
                {
                    task_queue.emplace(partition.right_vertices, task.level + 1);
                }
            }
            else
            {
                // Partition failed, fall back to direct
                std::cout << "  Multilevel partition invalid, using direct" << std::endl;
                Partition direct_partition_result = direct_partition(graph, task.vertices);
                direct_partition_result.print_stats();

                for (int v : direct_partition_result.separator_vertices)
                {
                    separator_stack.push_back(v);
                }

                if (!direct_partition_result.left_vertices.empty())
                {
                    task_queue.emplace(direct_partition_result.left_vertices, task.level + 1);
                }
                if (!direct_partition_result.right_vertices.empty())
                {
                    task_queue.emplace(direct_partition_result.right_vertices, task.level + 1);
                }
            }
        }
        else
        {
            // Use direct partitioning for smaller graphs
            std::cout << "  Using direct partitioning" << std::endl;
            Partition partition = direct_partition(graph, task.vertices);
            partition.print_stats();

            for (int v : partition.separator_vertices)
            {
                separator_stack.push_back(v);
            }

            if (!partition.left_vertices.empty())
            {
                task_queue.emplace(partition.left_vertices, task.level + 1);
            }
            if (!partition.right_vertices.empty())
            {
                task_queue.emplace(partition.right_vertices, task.level + 1);
            }
        }
    }

    // Add separators at the end in reverse order (typical nested dissection)
    std::reverse(separator_stack.begin(), separator_stack.end());
    for (int v : separator_stack)
    {
        h_ordering.push_back(v);
    }

    std::cout << "Final ordering: " << h_ordering.size() << " vertices" << std::endl;

    // Verify ordering completeness
    if (h_ordering.size() != graph.n_vertices)
    {
        std::cerr << "Warning: Incomplete ordering! Expected " << graph.n_vertices
                  << ", got " << h_ordering.size() << std::endl;
    }

    return h_ordering;
}

// Improved Multilevel Partitioning
// Replace the existing multilevel_partition function in nested_dissection.cu

// Improved Multilevel Partitioning
// Replace the existing multilevel_partition function in nested_dissection.cu

Partition ImprovedGPUNestedDissection::multilevel_partition(const Graph &original, const std::vector<int> &vertices)
{
    std::cout << "  Using multilevel partitioning" << std::endl;

    // Safety check: if subgraph is too small, use direct partitioning
    if (vertices.size() < 100)
    {
        std::cout << "  (Too small for multilevel, using direct)" << std::endl;
        return direct_partition(original, vertices);
    }

    try
    {
        // Extract subgraph
        Graph subgraph = extract_subgraph(original, vertices);

        // Coarsening phase - use heavy edge matching
        std::vector<Graph> graph_hierarchy;
        std::vector<std::vector<int>> vertex_mappings;

        // Store the initial subgraph
        graph_hierarchy.push_back(std::move(subgraph));

        // Coarsen until small enough or no more reduction possible
        int coarsening_levels = 0;
        const int max_coarsening_levels = 10;

        while (graph_hierarchy.back().n_vertices > 50 && coarsening_levels < max_coarsening_levels)
        {
            auto [coarse_graph, mapping] = coarsen_graph(graph_hierarchy.back());

            // Check if we got sufficient reduction
            double reduction_ratio = (double)coarse_graph.n_vertices / graph_hierarchy.back().n_vertices;
            if (reduction_ratio > 0.85)
            {
                // Not enough reduction, stop coarsening
                std::cout << "  Insufficient reduction (" << (reduction_ratio * 100)
                          << "%), stopping coarsening" << std::endl;
                break;
            }

            vertex_mappings.push_back(mapping);
            graph_hierarchy.push_back(std::move(coarse_graph));
            coarsening_levels++;

            std::cout << "  Coarsening level " << coarsening_levels
                      << ": " << graph_hierarchy.back().n_vertices << " vertices" << std::endl;
        }

        // Initial partition of coarsest graph
        Partition coarse_partition = initial_partition(graph_hierarchy.back());
        std::cout << "  Initial partition on coarsest graph completed" << std::endl;

        // Uncoarsening and refinement phase
        for (int i = graph_hierarchy.size() - 2; i >= 0; i--)
        {
            std::cout << "  Projecting to level " << i
                      << " (" << graph_hierarchy[i].n_vertices << " vertices)" << std::endl;
            std::cout << "  Current partition sizes: L=" << coarse_partition.left_vertices.size()
                      << " R=" << coarse_partition.right_vertices.size()
                      << " S=" << coarse_partition.separator_vertices.size() << std::endl;

            // Safety check before projection
            if (i < vertex_mappings.size())
            {
                std::cout << "  Vertex mapping size: " << vertex_mappings[i].size() << std::endl;
                coarse_partition = project_and_refine(graph_hierarchy[i], coarse_partition,
                                                      vertex_mappings[i]);
            }
            else
            {
                std::cout << "  ERROR: Invalid mapping index " << i << std::endl;
                break;
            }
        }

        // Map back to original vertex IDs
        Partition final_partition;
        std::cout << "  Mapping back to original vertices (size: " << vertices.size() << ")" << std::endl;

        for (int v : coarse_partition.left_vertices)
        {
            if (v >= 0 && v < vertices.size())
            {
                final_partition.left_vertices.push_back(vertices[v]);
            }
            else
            {
                std::cout << "  WARNING: Invalid left vertex index " << v << std::endl;
            }
        }
        for (int v : coarse_partition.right_vertices)
        {
            if (v >= 0 && v < vertices.size())
            {
                final_partition.right_vertices.push_back(vertices[v]);
            }
            else
            {
                std::cout << "  WARNING: Invalid right vertex index " << v << std::endl;
            }
        }
        for (int v : coarse_partition.separator_vertices)
        {
            if (v >= 0 && v < vertices.size())
            {
                final_partition.separator_vertices.push_back(vertices[v]);
            }
            else
            {
                std::cout << "  WARNING: Invalid separator vertex index " << v << std::endl;
            }
        }

        final_partition.edge_cut = coarse_partition.edge_cut;
        final_partition.balance_ratio = coarse_partition.balance_ratio;

        std::cout << "  Multilevel partitioning completed successfully" << std::endl;
        return final_partition;
    }
    catch (const std::exception &e)
    {
        std::cout << "  Multilevel partitioning failed: " << e.what() << std::endl;
        std::cout << "  Falling back to direct partitioning" << std::endl;
        return direct_partition(original, vertices);
    }
}

// Fallback direct partitioning (improved geometric + graph-based)
Partition ImprovedGPUNestedDissection::direct_partition(const Graph &graph, const std::vector<int> &vertices)
{
    std::cout << "  Using direct partitioning" << std::endl;

    Graph subgraph = extract_subgraph(graph, vertices);
    return graph_based_partition(subgraph, vertices);
}

// Heavy edge matching for coarsening (currently simplified to avoid segfaults)

// Replace the existing coarsen_graph function in nested_dissection.cu
std::pair<Graph, std::vector<int>> ImprovedGPUNestedDissection::coarsen_graph(const Graph &graph)
{
    const int n = graph.n_vertices;
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Allocate device memory for matching
    int *d_proposals, *d_coarse_vertex_count, *d_coarse_edge_count;
    cudaMalloc(&d_proposals, n * sizeof(int));
    cudaMalloc(&d_coarse_vertex_count, sizeof(int));
    cudaMalloc(&d_coarse_edge_count, sizeof(int));

    // Initialize matching arrays
    initialize_matching_arrays<<<grid, block>>>(d_match, d_temp_array, n);
    cudaDeviceSynchronize();

    // Phase 1: Each vertex proposes to its heaviest neighbor
    heavy_edge_matching_phase1<<<grid, block>>>(
        graph.row_ptr, graph.col_idx, graph.edge_weights,
        d_match, d_proposals, d_rand_states, n);
    cudaDeviceSynchronize();

    // Phase 2: Create matches from mutual proposals
    heavy_edge_matching_phase2<<<grid, block>>>(d_proposals, d_match, n);
    cudaDeviceSynchronize();

    // Initialize counters
    cudaMemset(d_coarse_vertex_count, 0, sizeof(int));
    cudaMemset(d_coarse_edge_count, 0, sizeof(int));

    // Compute vertex mapping and count coarse vertices
    compute_coarse_vertex_mapping<<<grid, block>>>(d_match, d_vertex_map, d_coarse_vertex_count, n);
    cudaDeviceSynchronize();

    // Get coarse vertex count
    int coarse_n;
    cudaMemcpy(&coarse_n, d_coarse_vertex_count, sizeof(int), cudaMemcpyDeviceToHost);

    if (coarse_n <= 0 || coarse_n >= n)
    {
        // Coarsening failed, return original graph
        std::cout << "  Coarsening failed, returning original graph" << std::endl;
        cudaFree(d_proposals);
        cudaFree(d_coarse_vertex_count);
        cudaFree(d_coarse_edge_count);

        // Create identity mapping
        std::vector<int> identity_map(n);
        std::iota(identity_map.begin(), identity_map.end(), 0);

        // Deep copy the original graph
        Graph copy_graph(n, graph.n_edges);
        cudaMemcpy(copy_graph.row_ptr, graph.row_ptr, (n + 1) * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(copy_graph.col_idx, graph.col_idx, graph.n_edges * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(copy_graph.edge_weights, graph.edge_weights, graph.n_edges * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(copy_graph.vertex_weights, graph.vertex_weights, n * sizeof(int), cudaMemcpyDeviceToDevice);

        return {std::move(copy_graph), identity_map};
    }

    // Count edges in coarse graph
    count_coarse_edges<<<grid, block>>>(graph.row_ptr, graph.col_idx, d_vertex_map, d_coarse_edge_count, n);
    cudaDeviceSynchronize();

    int coarse_edges;
    cudaMemcpy(&coarse_edges, d_coarse_edge_count, sizeof(int), cudaMemcpyDeviceToHost);

    // Allocate memory for coarse graph
    int *d_new_row_ptr, *d_edge_positions, *d_temp_neighbors, *d_temp_weights;
    cudaMalloc(&d_new_row_ptr, (coarse_n + 1) * sizeof(int));
    cudaMalloc(&d_edge_positions, coarse_n * sizeof(int));
    cudaMalloc(&d_temp_neighbors, coarse_edges * sizeof(int));
    cudaMalloc(&d_temp_weights, coarse_edges * sizeof(int));

    // Initialize arrays
    cudaMemset(d_new_row_ptr, 0, (coarse_n + 1) * sizeof(int));
    cudaMemset(d_edge_positions, 0, coarse_n * sizeof(int));

    // First pass: count edges per coarse vertex
    int *d_degree_count;
    cudaMalloc(&d_degree_count, coarse_n * sizeof(int));
    cudaMemset(d_degree_count, 0, coarse_n * sizeof(int));

    // Count degrees for each coarse vertex
    count_coarse_vertex_degrees<<<grid, block>>>(
        graph.row_ptr, graph.col_idx, d_vertex_map, d_degree_count, n);
    cudaDeviceSynchronize();

    // Compute prefix sum for row pointers using Thrust
    thrust::device_ptr<int> degree_ptr(d_degree_count);
    thrust::device_ptr<int> row_ptr(d_new_row_ptr + 1);
    thrust::inclusive_scan(degree_ptr, degree_ptr + coarse_n, row_ptr);

    // Build coarse graph structure
    build_coarse_graph_structure<<<grid, block>>>(
        graph.row_ptr, graph.col_idx, graph.edge_weights, d_vertex_map, n, coarse_n,
        d_new_row_ptr, d_edge_positions, d_temp_neighbors, d_temp_weights, d_temp_neighbors);
    cudaDeviceSynchronize();

    // Create final coarse graph
    Graph coarse_graph(coarse_n, coarse_edges);

    // Copy row pointers
    cudaMemcpy(coarse_graph.row_ptr, d_new_row_ptr, (coarse_n + 1) * sizeof(int), cudaMemcpyDeviceToDevice);

    // Merge parallel edges and sort
    const dim3 coarse_grid((coarse_n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    merge_parallel_edges<<<coarse_grid, block>>>(
        coarse_graph.row_ptr, d_temp_neighbors, d_temp_weights,
        coarse_graph.col_idx, coarse_graph.edge_weights, coarse_n);
    cudaDeviceSynchronize();

    // Compute coarse vertex weights
    cudaMemset(coarse_graph.vertex_weights, 0, coarse_n * sizeof(int));
    compute_vertex_weights_coarse<<<grid, block>>>(
        d_match, graph.vertex_weights, coarse_graph.vertex_weights, d_vertex_map, n);
    cudaDeviceSynchronize();

    // Copy vertex mapping to host
    std::vector<int> vertex_mapping(n);
    cudaMemcpy(vertex_mapping.data(), d_vertex_map, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_proposals);
    cudaFree(d_coarse_vertex_count);
    cudaFree(d_coarse_edge_count);
    cudaFree(d_new_row_ptr);
    cudaFree(d_edge_positions);
    cudaFree(d_temp_neighbors);
    cudaFree(d_temp_weights);
    cudaFree(d_degree_count);

    std::cout << "  GPU Coarsened from " << n << " to " << coarse_n << " vertices (reduction: "
              << (100.0 * (n - coarse_n) / n) << "%)" << std::endl;

    return {std::move(coarse_graph), vertex_mapping};
}

// Graph-based partitioning using BFS + edge cutting
Partition ImprovedGPUNestedDissection::graph_based_partition(const Graph &subgraph, const std::vector<int> &original_vertices)
{
    Partition result;
    int n = subgraph.n_vertices;

    if (n <= 2)
    {
        if (n == 1)
            result.left_vertices.push_back(original_vertices[0]);
        if (n == 2)
        {
            result.left_vertices.push_back(original_vertices[0]);
            result.right_vertices.push_back(original_vertices[1]);
        }
        return result;
    }

    // Extract subgraph adjacency from the actual subgraph structure
    std::vector<int> h_row_ptr(n + 1);
    std::vector<int> h_col_idx(subgraph.n_edges);
    std::vector<int> h_edge_weights(subgraph.n_edges);

    // Copy subgraph data to host for processing
    cudaMemcpy(h_row_ptr.data(), subgraph.row_ptr,
               (n + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_col_idx.data(), subgraph.col_idx,
               subgraph.n_edges * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_edge_weights.data(), subgraph.edge_weights,
               subgraph.n_edges * sizeof(int), cudaMemcpyDeviceToHost);

    // Build adjacency list from the actual subgraph
    std::vector<std::vector<int>> adj_list(n);
    for (int v = 0; v < n; v++)
    {
        for (int e = h_row_ptr[v]; e < h_row_ptr[v + 1]; e++)
        {
            int neighbor = h_col_idx[e];
            adj_list[v].push_back(neighbor);
        }
    }

    // Use BFS from multiple starting points to find good partition
    std::vector<int> best_partition(n, -1);
    int best_cut = INT_MAX;
    double best_balance = 1.0;

    // Try different starting vertices
    for (int start = 0; start < std::min(n, 4); start++)
    {
        std::vector<int> partition(n, -1);
        std::queue<int> bfs_queue;

        bfs_queue.push(start);
        partition[start] = 0;
        int left_count = 1;
        int target_size = n / 2;

        // BFS to grow one side
        while (!bfs_queue.empty() && left_count < target_size)
        {
            int v = bfs_queue.front();
            bfs_queue.pop();

            // Add unvisited neighbors to the same partition
            for (int neighbor : adj_list[v])
            {
                if (partition[neighbor] == -1 && left_count < target_size)
                {
                    partition[neighbor] = 0;
                    bfs_queue.push(neighbor);
                    left_count++;
                }
            }
        }

        // Assign remaining vertices to right partition
        for (int i = 0; i < n; i++)
        {
            if (partition[i] == -1)
            {
                partition[i] = 1;
            }
        }

        // Compute cut and balance
        int cut = 0;
        for (int i = 0; i < n; i++)
        {
            for (int neighbor : adj_list[i])
            {
                if (partition[i] != partition[neighbor])
                {
                    cut++;
                }
            }
        }
        cut /= 2; // Each edge counted twice

        int left_size = 0, right_size = 0;
        for (int i = 0; i < n; i++)
        {
            if (partition[i] == 0)
                left_size++;
            else
                right_size++;
        }

        double balance = abs(left_size - right_size) / (double)n;

        // Prefer better balance, then better cut
        if (balance < best_balance || (balance == best_balance && cut < best_cut))
        {
            best_cut = cut;
            best_balance = balance;
            best_partition = partition;
        }
    }

    // Find separator vertices (vertices with neighbors in both partitions)
    std::vector<bool> is_separator(n, false);
    for (int i = 0; i < n; i++)
    {
        bool has_left_neighbor = false, has_right_neighbor = false;

        for (int neighbor : adj_list[i])
        {
            if (best_partition[neighbor] == 0)
                has_left_neighbor = true;
            if (best_partition[neighbor] == 1)
                has_right_neighbor = true;
        }

        if (has_left_neighbor && has_right_neighbor)
        {
            is_separator[i] = true;
        }
    }

    // Build result - remove separator vertices from left/right partitions
    for (int i = 0; i < n; i++)
    {
        int orig_vertex = original_vertices[i];
        if (is_separator[i])
        {
            result.separator_vertices.push_back(orig_vertex);
        }
        else if (best_partition[i] == 0)
        {
            result.left_vertices.push_back(orig_vertex);
        }
        else
        {
            result.right_vertices.push_back(orig_vertex);
        }
    }

    result.edge_cut = best_cut;
    result.balance_ratio = best_balance;

    return result;
}

// Helper function to extract subgraph
Graph ImprovedGPUNestedDissection::extract_subgraph(const Graph &original, const std::vector<int> &vertices)
{
    // Create a mapping from old vertex IDs to new vertex IDs
    std::vector<int> vertex_map(original.n_vertices, -1);
    for (int i = 0; i < vertices.size(); i++)
    {
        vertex_map[vertices[i]] = i;
    }

    // Count edges in subgraph by examining original graph
    std::vector<int> h_row_ptr(original.n_vertices + 1);
    std::vector<int> h_col_idx(original.n_edges);
    std::vector<int> h_edge_weights(original.n_edges);

    // Copy original graph to host for processing
    cudaMemcpy(h_row_ptr.data(), original.row_ptr,
               (original.n_vertices + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_col_idx.data(), original.col_idx,
               original.n_edges * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_edge_weights.data(), original.edge_weights,
               original.n_edges * sizeof(int), cudaMemcpyDeviceToHost);

    // Count subgraph edges
    int sub_edges = 0;
    for (int v : vertices)
    {
        for (int e = h_row_ptr[v]; e < h_row_ptr[v + 1]; e++)
        {
            int neighbor = h_col_idx[e];
            if (vertex_map[neighbor] != -1)
            {
                sub_edges++;
            }
        }
    }

    // Create subgraph
    Graph subgraph(vertices.size(), sub_edges);
    std::vector<int> sub_row_ptr(vertices.size() + 1, 0);
    std::vector<int> sub_col_idx(sub_edges);
    std::vector<int> sub_edge_weights(sub_edges);
    std::vector<int> sub_vertex_weights(vertices.size(), 1);

    int edge_pos = 0;
    for (int i = 0; i < vertices.size(); i++)
    {
        int v = vertices[i];
        sub_row_ptr[i] = edge_pos;

        for (int e = h_row_ptr[v]; e < h_row_ptr[v + 1]; e++)
        {
            int neighbor = h_col_idx[e];
            if (vertex_map[neighbor] != -1)
            {
                sub_col_idx[edge_pos] = vertex_map[neighbor];
                sub_edge_weights[edge_pos] = h_edge_weights[e];
                edge_pos++;
            }
        }
    }
    sub_row_ptr[vertices.size()] = edge_pos;

    // Copy to device
    cudaMemcpy(subgraph.row_ptr, sub_row_ptr.data(),
               (vertices.size() + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(subgraph.col_idx, sub_col_idx.data(),
               sub_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(subgraph.edge_weights, sub_edge_weights.data(),
               sub_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(subgraph.vertex_weights, sub_vertex_weights.data(),
               vertices.size() * sizeof(int), cudaMemcpyHostToDevice);

    return subgraph;
}

// Initial partitioning for coarsest graph
Partition ImprovedGPUNestedDissection::initial_partition(const Graph &graph)
{
    // Improved initial partitioning for coarsest graph
    Partition result;

    // Copy graph data to host for initial partitioning
    std::vector<int> h_row_ptr(graph.n_vertices + 1);
    std::vector<int> h_col_idx(graph.n_edges);

    cudaMemcpy(h_row_ptr.data(), graph.row_ptr,
               (graph.n_vertices + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_col_idx.data(), graph.col_idx,
               graph.n_edges * sizeof(int), cudaMemcpyDeviceToHost);

    // Use BFS-based partitioning for better quality
    std::vector<int> partition(graph.n_vertices, -1);
    std::queue<int> bfs_queue;

    // Start from vertex with lowest degree (or random if all similar)
    int start_vertex = 0;
    int min_degree = h_row_ptr[1] - h_row_ptr[0];
    for (int i = 1; i < graph.n_vertices; i++)
    {
        int degree = h_row_ptr[i + 1] - h_row_ptr[i];
        if (degree < min_degree)
        {
            min_degree = degree;
            start_vertex = i;
        }
    }

    bfs_queue.push(start_vertex);
    partition[start_vertex] = 0;
    int left_count = 1;
    int target_size = graph.n_vertices / 2;

    // BFS to grow left partition
    while (!bfs_queue.empty() && left_count < target_size)
    {
        int v = bfs_queue.front();
        bfs_queue.pop();

        for (int e = h_row_ptr[v]; e < h_row_ptr[v + 1]; e++)
        {
            int neighbor = h_col_idx[e];
            if (partition[neighbor] == -1 && left_count < target_size)
            {
                partition[neighbor] = 0;
                bfs_queue.push(neighbor);
                left_count++;
            }
        }
    }

    // Assign remaining vertices to right partition
    for (int i = 0; i < graph.n_vertices; i++)
    {
        if (partition[i] == -1)
        {
            partition[i] = 1;
        }
    }

    // Find separator vertices
    std::vector<bool> is_separator(graph.n_vertices, false);
    for (int v = 0; v < graph.n_vertices; v++)
    {
        bool has_left = false, has_right = false;
        for (int e = h_row_ptr[v]; e < h_row_ptr[v + 1]; e++)
        {
            int neighbor = h_col_idx[e];
            if (partition[neighbor] == 0)
                has_left = true;
            if (partition[neighbor] == 1)
                has_right = true;
        }
        if (has_left && has_right)
        {
            is_separator[v] = true;
        }
    }

    // Build partition result
    for (int i = 0; i < graph.n_vertices; i++)
    {
        if (is_separator[i])
        {
            result.separator_vertices.push_back(i);
        }
        else if (partition[i] == 0)
        {
            result.left_vertices.push_back(i);
        }
        else
        {
            result.right_vertices.push_back(i);
        }
    }

    return result;
}

// Project partition to finer graph and refine
// Safer Project and Refine Function
// Replace the existing project_and_refine function in nested_dissection.cu

Partition ImprovedGPUNestedDissection::project_and_refine(const Graph &fine_graph, const Partition &coarse_partition,
                                                          const std::vector<int> &vertex_mapping)
{
    std::cout << "    Projecting partition from " << vertex_mapping.size()
              << " coarse vertices to " << fine_graph.n_vertices << " fine vertices" << std::endl;

    // Safety check
    if (vertex_mapping.size() != fine_graph.n_vertices)
    {
        std::cout << "    ERROR: Vertex mapping size mismatch!" << std::endl;
        // Return a simple bisection as fallback
        Partition fallback;
        for (int i = 0; i < fine_graph.n_vertices; i++)
        {
            if (i < fine_graph.n_vertices / 2)
            {
                fallback.left_vertices.push_back(i);
            }
            else
            {
                fallback.right_vertices.push_back(i);
            }
        }
        return fallback;
    }

    // Project partition to finer graph
    Partition projected;

    // Create a mapping from coarse vertex to partition assignment
    std::vector<int> coarse_assignment;
    int max_coarse_vertex = 0;

    // Find the maximum coarse vertex ID
    for (int mapping : vertex_mapping)
    {
        max_coarse_vertex = std::max(max_coarse_vertex, mapping);
    }

    coarse_assignment.resize(max_coarse_vertex + 1, -1);

    // Assign partitions to coarse vertices
    for (int v : coarse_partition.left_vertices)
    {
        if (v >= 0 && v < coarse_assignment.size())
        {
            coarse_assignment[v] = 0; // left partition
        }
    }
    for (int v : coarse_partition.right_vertices)
    {
        if (v >= 0 && v < coarse_assignment.size())
        {
            coarse_assignment[v] = 1; // right partition
        }
    }
    for (int v : coarse_partition.separator_vertices)
    {
        if (v >= 0 && v < coarse_assignment.size())
        {
            coarse_assignment[v] = 2; // separator
        }
    }

    // Project each fine vertex based on its coarse vertex assignment
    for (int fine_v = 0; fine_v < fine_graph.n_vertices; fine_v++)
    {
        int coarse_v = vertex_mapping[fine_v];

        if (coarse_v >= 0 && coarse_v < coarse_assignment.size())
        {
            int assignment = coarse_assignment[coarse_v];
            if (assignment == 0)
            {
                projected.left_vertices.push_back(fine_v);
            }
            else if (assignment == 1)
            {
                projected.right_vertices.push_back(fine_v);
            }
            else if (assignment == 2)
            {
                projected.separator_vertices.push_back(fine_v);
            }
            else
            {
                // Unassigned coarse vertex, default to left
                projected.left_vertices.push_back(fine_v);
            }
        }
        else
        {
            // Invalid coarse vertex mapping, default to left
            projected.left_vertices.push_back(fine_v);
        }
    }

    std::cout << "    Projected partition: L=" << projected.left_vertices.size()
              << " R=" << projected.right_vertices.size()
              << " S=" << projected.separator_vertices.size() << std::endl;

    // Apply basic local refinement (simplified and safer)
    return apply_local_refinement(fine_graph, projected);
}

// Apply local refinement to improve partition quality
Partition ImprovedGPUNestedDissection::apply_local_refinement(const Graph &graph, const Partition &initial_partition)
{
    Partition refined = initial_partition;

    // Copy graph to host for refinement
    std::vector<int> h_row_ptr(graph.n_vertices + 1);
    std::vector<int> h_col_idx(graph.n_edges);
    std::vector<int> h_edge_weights(graph.n_edges);

    cudaMemcpy(h_row_ptr.data(), graph.row_ptr,
               (graph.n_vertices + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_col_idx.data(), graph.col_idx,
               graph.n_edges * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_edge_weights.data(), graph.edge_weights,
               graph.n_edges * sizeof(int), cudaMemcpyDeviceToHost);

    // Create partition assignment array
    std::vector<int> partition(graph.n_vertices, -1);
    for (int v : refined.left_vertices)
        partition[v] = 0;
    for (int v : refined.right_vertices)
        partition[v] = 1;
    for (int v : refined.separator_vertices)
        partition[v] = -1;

    // Simple boundary refinement - move boundary vertices to minimize cut
    bool improved = true;
    int iterations = 0;
    const int max_iterations = 5;

    while (improved && iterations < max_iterations)
    {
        improved = false;
        iterations++;

        // Check each boundary vertex
        for (int v = 0; v < graph.n_vertices; v++)
        {
            if (partition[v] == -1)
                continue; // Skip separator vertices

            // Check if this vertex is on the partition boundary
            bool is_boundary = false;
            for (int e = h_row_ptr[v]; e < h_row_ptr[v + 1]; e++)
            {
                int neighbor = h_col_idx[e];
                if (partition[neighbor] != partition[v] && partition[neighbor] != -1)
                {
                    is_boundary = true;
                    break;
                }
            }

            if (!is_boundary)
                continue;

            // Compute gain of moving this vertex to the other partition
            int current_partition = partition[v];
            int other_partition = 1 - current_partition;

            int internal_weight = 0, external_weight = 0;

            for (int e = h_row_ptr[v]; e < h_row_ptr[v + 1]; e++)
            {
                int neighbor = h_col_idx[e];
                int weight = h_edge_weights[e];

                if (partition[neighbor] == current_partition)
                {
                    internal_weight += weight;
                }
                else if (partition[neighbor] == other_partition)
                {
                    external_weight += weight;
                }
            }

            int gain = external_weight - internal_weight;

            // Move vertex if gain is positive (reduces cut)
            if (gain > 0)
            {
                partition[v] = other_partition;
                improved = true;
            }
        }
    }

    // Rebuild partition from refined assignment
    refined.left_vertices.clear();
    refined.right_vertices.clear();
    // Keep existing separator vertices

    for (int v = 0; v < graph.n_vertices; v++)
    {
        if (partition[v] == 0)
        {
            refined.left_vertices.push_back(v);
        }
        else if (partition[v] == 1)
        {
            refined.right_vertices.push_back(v);
        }
    }

    return refined;
}

// Compute partition cut weight
int ImprovedGPUNestedDissection::compute_partition_cut(const Graph &graph, const std::vector<int> &partition)
{
    // Copy graph data to host
    std::vector<int> h_row_ptr(graph.n_vertices + 1);
    std::vector<int> h_col_idx(graph.n_edges);
    std::vector<int> h_edge_weights(graph.n_edges);

    cudaMemcpy(h_row_ptr.data(), graph.row_ptr,
               (graph.n_vertices + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_col_idx.data(), graph.col_idx,
               graph.n_edges * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_edge_weights.data(), graph.edge_weights,
               graph.n_edges * sizeof(int), cudaMemcpyDeviceToHost);

    int cut = 0;
    for (int v = 0; v < graph.n_vertices; v++)
    {
        for (int e = h_row_ptr[v]; e < h_row_ptr[v + 1]; e++)
        {
            int neighbor = h_col_idx[e];
            if (v < neighbor && partition[v] != partition[neighbor] &&
                partition[v] != -1 && partition[neighbor] != -1)
            {
                cut += h_edge_weights[e];
            }
        }
    }
    return cut;
}

// Identify separator vertices based on partition
void ImprovedGPUNestedDissection::identify_separator_vertices(const Graph &graph, const std::vector<int> &partition,
                                                              std::vector<bool> &is_separator)
{
    // Copy graph data to host
    std::vector<int> h_row_ptr(graph.n_vertices + 1);
    std::vector<int> h_col_idx(graph.n_edges);

    cudaMemcpy(h_row_ptr.data(), graph.row_ptr,
               (graph.n_vertices + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_col_idx.data(), graph.col_idx,
               graph.n_edges * sizeof(int), cudaMemcpyDeviceToHost);

    for (int v = 0; v < graph.n_vertices; v++)
    {
        bool has_left_neighbor = false, has_right_neighbor = false;

        for (int e = h_row_ptr[v]; e < h_row_ptr[v + 1]; e++)
        {
            int neighbor = h_col_idx[e];
            if (partition[neighbor] == 0)
                has_left_neighbor = true;
            if (partition[neighbor] == 1)
                has_right_neighbor = true;
        }

        is_separator[v] = has_left_neighbor && has_right_neighbor;
    }
}

// Static method to create test grid graph
Graph ImprovedGPUNestedDissection::create_test_grid_graph(int grid_size)
{
    const int n_vertices = grid_size * grid_size;

    std::vector<std::vector<std::pair<int, int>>> adj_list(n_vertices);

    // Create 2D grid with edge weights
    for (int i = 0; i < grid_size; i++)
    {
        for (int j = 0; j < grid_size; j++)
        {
            int v = i * grid_size + j;

            // Right neighbor
            if (j < grid_size - 1)
            {
                int neighbor = i * grid_size + (j + 1);
                int weight = 1 + (i + j) % 3; // Varying edge weights
                adj_list[v].push_back({neighbor, weight});
                adj_list[neighbor].push_back({v, weight});
            }

            // Bottom neighbor
            if (i < grid_size - 1)
            {
                int neighbor = (i + 1) * grid_size + j;
                int weight = 1 + (i * j) % 3; // Varying edge weights
                adj_list[v].push_back({neighbor, weight});
                adj_list[neighbor].push_back({v, weight});
            }
        }
    }

    // Count total edges
    int total_edges = 0;
    for (const auto &neighbors : adj_list)
    {
        total_edges += neighbors.size();
    }

    // Create CSR format
    Graph graph(n_vertices, total_edges);
    std::vector<int> h_row_ptr(n_vertices + 1, 0);
    std::vector<int> h_col_idx(total_edges);
    std::vector<int> h_edge_weights(total_edges);
    std::vector<int> h_vertex_weights(n_vertices, 1);

    int edge_pos = 0;
    for (int v = 0; v < n_vertices; v++)
    {
        h_row_ptr[v] = edge_pos;
        for (auto [neighbor, weight] : adj_list[v])
        {
            h_col_idx[edge_pos] = neighbor;
            h_edge_weights[edge_pos] = weight;
            edge_pos++;
        }
    }
    h_row_ptr[n_vertices] = edge_pos;

    // Copy to device
    cudaMemcpy(graph.row_ptr, h_row_ptr.data(), (n_vertices + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(graph.col_idx, h_col_idx.data(), total_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(graph.edge_weights, h_edge_weights.data(), total_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(graph.vertex_weights, h_vertex_weights.data(), n_vertices * sizeof(int), cudaMemcpyHostToDevice);

    return graph;
}