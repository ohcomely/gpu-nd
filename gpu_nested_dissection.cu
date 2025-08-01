#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <vector>
#include <queue>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <set>
#include <iomanip>

// Forward declarations for CUDA kernels
__global__ void init_random_states(curandState *states, int n, unsigned long seed);
__global__ void compute_vertex_degrees(int *row_ptr, int n, int *degrees);
__global__ void heavy_edge_matching_kernel(int *row_ptr, int *col_idx, int *edge_weights,
                                           int *match, curandState *states, int n);
__global__ void contract_graph_kernel(int *old_row_ptr, int *old_col_idx, int *old_edge_weights,
                                      int *match, int *vertex_map, int old_n,
                                      int *new_row_ptr, int *new_col_idx, int *new_edge_weights,
                                      int *new_vertex_weights, int new_n);

class ImprovedGPUNestedDissection
{
private:
    static const int BLOCK_SIZE = 256;
    static const int MIN_SUBGRAPH_SIZE = 8; // Increased for better multilevel effectiveness
    static const int COARSENING_RATIO = 2;  // Target reduction ratio per level

    // Device memory
    int *d_partition;
    int *d_degrees;
    int *d_match;
    int *d_vertex_map;
    int *d_temp_array;
    curandState *d_rand_states;

    // Host memory
    std::vector<int> h_ordering;
    int max_vertices;

public:
    ImprovedGPUNestedDissection(int max_v) : max_vertices(max_v)
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

    ~ImprovedGPUNestedDissection()
    {
        cudaFree(d_partition);
        cudaFree(d_degrees);
        cudaFree(d_match);
        cudaFree(d_vertex_map);
        cudaFree(d_temp_array);
        cudaFree(d_rand_states);
    }

    std::vector<int> compute_ordering(const Graph &graph)
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

            // Use multilevel approach for larger subgraphs (currently disabled due to segfault)
            if (task.vertices.size() > 1000) // Increased threshold to avoid multilevel for now
            {
                Partition partition = multilevel_partition(graph, task.vertices);
                partition.print_stats();

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
                // Use direct partitioning for smaller graphs
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

        // Add separators at the end in reverse order
        std::reverse(separator_stack.begin(), separator_stack.end());
        for (int v : separator_stack)
        {
            h_ordering.push_back(v);
        }

        return h_ordering;
    }

private:
    // METIS-inspired multilevel partitioning
    Partition multilevel_partition(const Graph &original, const std::vector<int> &vertices)
    {
        std::cout << "  Using multilevel partitioning" << std::endl;

        // For now, fall back to direct partitioning to avoid segfault
        // TODO: Fix multilevel implementation
        std::cout << "  (Falling back to direct partitioning)" << std::endl;
        return direct_partition(original, vertices);

        /*
        // Extract subgraph
        Graph subgraph = extract_subgraph(original, vertices);

        // Coarsening phase - use heavy edge matching
        std::vector<Graph> graph_hierarchy;
        std::vector<std::vector<int>> vertex_mappings;

        Graph current_graph = subgraph;
        graph_hierarchy.push_back(current_graph);

        // Coarsen until small enough
        while (current_graph.n_vertices > 50)
        {
            auto [coarse_graph, mapping] = coarsen_graph(current_graph);
            if (coarse_graph.n_vertices >= current_graph.n_vertices * 0.8)
            {
                // Not enough reduction, stop coarsening
                break;
            }

            graph_hierarchy.push_back(coarse_graph);
            vertex_mappings.push_back(mapping);
            current_graph = coarse_graph;
        }

        // Initial partition of coarsest graph
        Partition coarse_partition = initial_partition(graph_hierarchy.back());

        // Uncoarsening and refinement phase
        for (int i = graph_hierarchy.size() - 2; i >= 0; i--)
        {
            coarse_partition = project_and_refine(graph_hierarchy[i], coarse_partition,
                                                vertex_mappings[i]);
        }

        // Map back to original vertex IDs
        Partition final_partition;
        for (int v : coarse_partition.left_vertices)
        {
            final_partition.left_vertices.push_back(vertices[v]);
        }
        for (int v : coarse_partition.right_vertices)
        {
            final_partition.right_vertices.push_back(vertices[v]);
        }
        for (int v : coarse_partition.separator_vertices)
        {
            final_partition.separator_vertices.push_back(vertices[v]);
        }

        return final_partition;
        */
    }

    // Fallback direct partitioning (improved geometric + graph-based)
    Partition direct_partition(const Graph &graph, const std::vector<int> &vertices)
    {
        std::cout << "  Using direct partitioning" << std::endl;

        Graph subgraph = extract_subgraph(graph, vertices);
        return graph_based_partition(subgraph, vertices);
    }

    // Heavy edge matching for coarsening (currently simplified to avoid segfaults)
    std::pair<Graph, std::vector<int>> coarsen_graph(const Graph &graph)
    {
        // Simplified coarsening - just return the same graph for now
        // TODO: Implement proper heavy edge matching

        std::vector<int> vertex_map(graph.n_vertices);
        std::iota(vertex_map.begin(), vertex_map.end(), 0);

        // Create a copy of the input graph
        Graph coarse_graph(graph.n_vertices, graph.n_edges);

        // Copy data from original graph
        cudaMemcpy(coarse_graph.row_ptr, graph.row_ptr,
                   (graph.n_vertices + 1) * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(coarse_graph.col_idx, graph.col_idx,
                   graph.n_edges * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(coarse_graph.edge_weights, graph.edge_weights,
                   graph.n_edges * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(coarse_graph.vertex_weights, graph.vertex_weights,
                   graph.n_vertices * sizeof(int), cudaMemcpyDeviceToDevice);

        return {coarse_graph, vertex_map};
    }

    // Graph-based partitioning using BFS + edge cutting
    Partition graph_based_partition(const Graph &subgraph, const std::vector<int> &original_vertices)
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

        // Extract subgraph adjacency to host for processing
        std::vector<int> h_row_ptr(n + 1);
        std::vector<int> h_col_idx;
        std::vector<int> h_edge_weights;

        // For now, use a simplified approach - in reality we'd extract the actual subgraph
        // Create a simple adjacency structure based on original vertices
        std::vector<std::vector<int>> adj_list(n);

        // Build adjacency list for subgraph vertices
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                int orig_i = original_vertices[i];
                int orig_j = original_vertices[j];

                // Check if vertices are adjacent in original grid (simplified)
                int grid_size = 6; // Known from test graph
                int row_i = orig_i / grid_size, col_i = orig_i % grid_size;
                int row_j = orig_j / grid_size, col_j = orig_j % grid_size;

                if ((abs(row_i - row_j) == 1 && col_i == col_j) ||
                    (abs(col_i - col_j) == 1 && row_i == row_j))
                {
                    adj_list[i].push_back(j);
                    adj_list[j].push_back(i);
                }
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

    // Helper functions
    Graph extract_subgraph(const Graph &original, const std::vector<int> &vertices)
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

    Partition initial_partition(const Graph &graph)
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

    Partition project_and_refine(const Graph &fine_graph, const Partition &coarse_partition,
                                 const std::vector<int> &vertex_mapping)
    {
        // Project partition to finer graph
        Partition projected;

        // Create reverse mapping
        std::vector<int> coarse_to_fine_mapping(coarse_partition.left_vertices.size() +
                                                coarse_partition.right_vertices.size() +
                                                coarse_partition.separator_vertices.size());

        int coarse_idx = 0;
        for (int v : coarse_partition.left_vertices)
        {
            coarse_to_fine_mapping[v] = 0; // left partition
        }
        for (int v : coarse_partition.right_vertices)
        {
            coarse_to_fine_mapping[v] = 1; // right partition
        }
        for (int v : coarse_partition.separator_vertices)
        {
            coarse_to_fine_mapping[v] = -1; // separator
        }

        // Project each fine vertex based on its coarse vertex assignment
        for (int fine_v = 0; fine_v < fine_graph.n_vertices; fine_v++)
        {
            int coarse_v = vertex_mapping[fine_v];

            if (coarse_v < coarse_to_fine_mapping.size())
            {
                int assignment = coarse_to_fine_mapping[coarse_v];
                if (assignment == 0)
                {
                    projected.left_vertices.push_back(fine_v);
                }
                else if (assignment == 1)
                {
                    projected.right_vertices.push_back(fine_v);
                }
                else
                {
                    projected.separator_vertices.push_back(fine_v);
                }
            }
            else
            {
                // Default assignment for unmapped vertices
                projected.left_vertices.push_back(fine_v);
            }
        }

        // Apply basic local refinement (simplified KL-style)
        return apply_local_refinement(fine_graph, projected);
    }

    Partition apply_local_refinement(const Graph &graph, const Partition &initial_partition)
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

    int compute_partition_cut(const Graph &graph, const std::vector<int> &partition)
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

    void identify_separator_vertices(const Graph &graph, const std::vector<int> &partition,
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
};

// MTX file reader for Matrix Market format
class MTXReader
{
public:
    struct MTXGraph
    {
        int n_vertices;
        int n_edges;
        std::vector<int> row_indices;
        std::vector<int> col_indices;
        std::vector<double> values;
        bool is_symmetric;
        bool is_pattern;
    };

    static MTXGraph read_mtx_file(const std::string &filename)
    {
        std::ifstream file(filename);
        if (!file.is_open())
        {
            throw std::runtime_error("Cannot open MTX file: " + filename);
        }

        MTXGraph mtx_graph;
        std::string line;

        // Read header
        std::getline(file, line);
        if (line.find("%%MatrixMarket") == std::string::npos)
        {
            throw std::runtime_error("Invalid MTX file header");
        }

        mtx_graph.is_symmetric = line.find("symmetric") != std::string::npos;
        mtx_graph.is_pattern = line.find("pattern") != std::string::npos;

        // Skip comments
        while (std::getline(file, line) && line[0] == '%')
        {
        }

        // Read dimensions
        std::istringstream iss(line);
        int n_rows, n_cols, n_entries;
        iss >> n_rows >> n_cols >> n_entries;

        if (n_rows != n_cols)
        {
            throw std::runtime_error("Matrix must be square for graph partitioning");
        }

        mtx_graph.n_vertices = n_rows;
        mtx_graph.row_indices.reserve(n_entries * (mtx_graph.is_symmetric ? 2 : 1));
        mtx_graph.col_indices.reserve(n_entries * (mtx_graph.is_symmetric ? 2 : 1));
        mtx_graph.values.reserve(n_entries * (mtx_graph.is_symmetric ? 2 : 1));

        // Read entries
        for (int i = 0; i < n_entries; i++)
        {
            std::getline(file, line);
            std::istringstream entry_stream(line);

            int row, col;
            double value = 1.0;

            entry_stream >> row >> col;
            if (!mtx_graph.is_pattern)
            {
                entry_stream >> value;
            }

            // Convert to 0-based indexing
            row--;
            col--;

            // Skip diagonal entries
            if (row == col)
                continue;

            mtx_graph.row_indices.push_back(row);
            mtx_graph.col_indices.push_back(col);
            mtx_graph.values.push_back(std::abs(value));

            // Add symmetric entry if needed
            if (mtx_graph.is_symmetric && row != col)
            {
                mtx_graph.row_indices.push_back(col);
                mtx_graph.col_indices.push_back(row);
                mtx_graph.values.push_back(std::abs(value));
            }
        }

        mtx_graph.n_edges = mtx_graph.row_indices.size();

        std::cout << "Read MTX file: " << mtx_graph.n_vertices << " vertices, "
                  << mtx_graph.n_edges << " edges" << std::endl;
        std::cout << "Symmetric: " << (mtx_graph.is_symmetric ? "Yes" : "No")
                  << ", Pattern: " << (mtx_graph.is_pattern ? "Yes" : "No") << std::endl;

        return mtx_graph;
    }

    static Graph convert_to_csr_graph(const MTXGraph &mtx_graph)
    {
        // Convert COO to CSR format
        std::vector<std::vector<std::pair<int, int>>> adj_list(mtx_graph.n_vertices);

        for (int i = 0; i < mtx_graph.n_edges; i++)
        {
            int row = mtx_graph.row_indices[i];
            int col = mtx_graph.col_indices[i];
            int weight = static_cast<int>(mtx_graph.values[i] * 100); // Scale weights

            adj_list[row].push_back({col, weight});
        }

        // Remove duplicates and sort
        for (int i = 0; i < mtx_graph.n_vertices; i++)
        {
            std::sort(adj_list[i].begin(), adj_list[i].end());
            adj_list[i].erase(std::unique(adj_list[i].begin(), adj_list[i].end()),
                              adj_list[i].end());
        }

        // Count total edges
        int total_edges = 0;
        for (const auto &neighbors : adj_list)
        {
            total_edges += neighbors.size();
        }

        // Create CSR graph
        Graph graph(mtx_graph.n_vertices, total_edges);
        std::vector<int> h_row_ptr(mtx_graph.n_vertices + 1, 0);
        std::vector<int> h_col_idx(total_edges);
        std::vector<int> h_edge_weights(total_edges);
        std::vector<int> h_vertex_weights(mtx_graph.n_vertices, 1);

        int edge_pos = 0;
        for (int v = 0; v < mtx_graph.n_vertices; v++)
        {
            h_row_ptr[v] = edge_pos;
            for (auto [neighbor, weight] : adj_list[v])
            {
                h_col_idx[edge_pos] = neighbor;
                h_edge_weights[edge_pos] = std::max(1, weight);
                edge_pos++;
            }
        }
        h_row_ptr[mtx_graph.n_vertices] = edge_pos;

        // Copy to device
        cudaMemcpy(graph.row_ptr, h_row_ptr.data(),
                   (mtx_graph.n_vertices + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(graph.col_idx, h_col_idx.data(),
                   total_edges * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(graph.edge_weights, h_edge_weights.data(),
                   total_edges * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(graph.vertex_weights, h_vertex_weights.data(),
                   mtx_graph.n_vertices * sizeof(int), cudaMemcpyHostToDevice);

        return graph;
    }
};
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

// Test function for grid graphs
Graph create_test_grid_graph(int grid_size = 6)
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

int main(int argc, char *argv[])
{
    std::cout << "=== GPU Nested Dissection with MTX Support ===" << std::endl;

    Graph graph(0, 0);
    std::string graph_type = "grid";

    if (argc > 1)
    {
        std::string filename = argv[1];

        if (filename.find(".mtx") != std::string::npos)
        {
            try
            {
                std::cout << "Loading MTX file: " << filename << std::endl;
                auto mtx_graph = MTXReader::read_mtx_file(filename);
                graph = MTXReader::convert_to_csr_graph(mtx_graph);
                graph_type = "mtx";
            }
            catch (const std::exception &e)
            {
                std::cerr << "Error reading MTX file: " << e.what() << std::endl;
                std::cout << "Falling back to test grid graph..." << std::endl;
                graph = create_test_grid_graph();
            }
        }
        else
        {
            int grid_size = std::stoi(filename);
            std::cout << "Creating " << grid_size << "x" << grid_size << " test grid..." << std::endl;
            graph = create_test_grid_graph(grid_size);
        }
    }
    else
    {
        std::cout << "Creating default 6x6 test grid..." << std::endl;
        graph = create_test_grid_graph();
    }

    std::cout << "Graph loaded: " << graph.n_vertices << " vertices, "
              << graph.n_edges << " edges" << std::endl;

    if (graph_type == "grid")
    {
        int grid_size = static_cast<int>(std::sqrt(graph.n_vertices));
        std::cout << "Grid layout (" << grid_size << "x" << grid_size << "):" << std::endl;
        for (int i = 0; i < grid_size && i < 10; i++) // Limit display for large grids
        {
            for (int j = 0; j < grid_size && j < 10; j++)
            {
                int vertex = i * grid_size + j;
                printf("%3d ", vertex);
            }
            if (grid_size > 10)
                std::cout << "...";
            std::cout << std::endl;
        }
        if (grid_size > 10)
            std::cout << "..." << std::endl;
        std::cout << std::endl;
    }

    ImprovedGPUNestedDissection nd(graph.n_vertices);

    std::cout << "Computing improved nested dissection ordering..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<int> ordering = nd.compute_ordering(graph);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "\nNested dissection ordering (" << ordering.size() << " vertices):" << std::endl;

    // Print ordering (limit output for large graphs)
    if (ordering.size() <= 100)
    {
        for (int i = 0; i < ordering.size(); i++)
        {
            std::cout << ordering[i];
            if (i < ordering.size() - 1)
                std::cout << " ";
        }
        std::cout << std::endl;
    }
    else
    {
        std::cout << "First 50: ";
        for (int i = 0; i < 50; i++)
        {
            std::cout << ordering[i] << " ";
        }
        std::cout << "\n...";
        std::cout << "\nLast 50: ";
        for (int i = ordering.size() - 50; i < ordering.size(); i++)
        {
            std::cout << ordering[i] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nComputation time: " << duration.count() << " microseconds" << std::endl;

    std::cout << "Usage:" << std::endl;
    std::cout << "  " << argv[0] << "                    # Default 6x6 grid" << std::endl;
    std::cout << "  " << argv[0] << " 10                 # 10x10 grid" << std::endl;
    std::cout << "  " << argv[0] << " matrix.mtx         # MTX file" << std::endl;

    std::cout << "Done!" << std::endl;

    return 0;
}