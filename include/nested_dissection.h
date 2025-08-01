#pragma once

#include "structs.h"
#include <curand_kernel.h>
#include <vector>
#include <queue>

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
    /**
     * Constructor - allocates GPU memory and initializes random states
     * @param max_v Maximum number of vertices to support
     */
    explicit ImprovedGPUNestedDissection(int max_v);

    /**
     * Destructor - frees GPU memory
     */
    ~ImprovedGPUNestedDissection();

    /**
     * Compute nested dissection ordering for the given graph
     * @param graph Input graph in CSR format
     * @return Vector containing the vertex ordering
     */
    std::vector<int> compute_ordering(const Graph &graph);

    /**
     * Create a test grid graph for testing purposes
     * @param grid_size Size of the grid (grid_size x grid_size vertices)
     * @return Graph object representing the grid
     */
    static Graph create_test_grid_graph(int grid_size = 6);

private:
    // Core partitioning methods
    Partition multilevel_partition(const Graph &original, const std::vector<int> &vertices);
    Partition direct_partition(const Graph &graph, const std::vector<int> &vertices);

    // Graph coarsening methods
    std::pair<Graph, std::vector<int>> coarsen_graph(const Graph &graph);

    // Partitioning algorithms
    Partition graph_based_partition(const Graph &subgraph, const std::vector<int> &original_vertices);
    Partition initial_partition(const Graph &graph);

    // Refinement methods
    Partition project_and_refine(const Graph &fine_graph, const Partition &coarse_partition,
                                 const std::vector<int> &vertex_mapping);
    Partition apply_local_refinement(const Graph &graph, const Partition &initial_partition);

    // Utility methods
    Graph extract_subgraph(const Graph &original, const std::vector<int> &vertices);
    int compute_partition_cut(const Graph &graph, const std::vector<int> &partition);
    void identify_separator_vertices(const Graph &graph, const std::vector<int> &partition,
                                     std::vector<bool> &is_separator);

    // Delete copy constructor and assignment operator
    ImprovedGPUNestedDissection(const ImprovedGPUNestedDissection &) = delete;
    ImprovedGPUNestedDissection &operator=(const ImprovedGPUNestedDissection &) = delete;
};