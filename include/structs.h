#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cassert>

// Forward declarations for CUDA kernels
__global__ void init_random_states(curandState *states, int n, unsigned long seed);
__global__ void compute_vertex_degrees(int *row_ptr, int n, int *degrees);
__global__ void heavy_edge_matching_kernel(int *row_ptr, int *col_idx, int *edge_weights,
                                           int *match, curandState *states, int n);
__global__ void contract_graph_kernel(int *old_row_ptr, int *old_col_idx, int *old_edge_weights,
                                      int *match, int *vertex_map, int old_n,
                                      int *new_row_ptr, int *new_col_idx, int *new_edge_weights,
                                      int *new_vertex_weights, int new_n);

// Improved CSR graph representation with edge weights
struct Graph
{
    int n_vertices;
    int n_edges;
    int *row_ptr;        // Size: n_vertices + 1
    int *col_idx;        // Size: n_edges
    int *edge_weights;   // Size: n_edges
    int *vertex_weights; // Size: n_vertices

    Graph(int n_v, int n_e);
    ~Graph();

    // Move constructor and assignment operator
    Graph(Graph &&other) noexcept;
    Graph &operator=(Graph &&other) noexcept;

    // Delete copy constructor and copy assignment to prevent accidental copies
    Graph(const Graph &) = delete;
    Graph &operator=(const Graph &) = delete;
};

// Enhanced partition result with quality metrics
struct Partition
{
    std::vector<int> left_vertices;
    std::vector<int> right_vertices;
    std::vector<int> separator_vertices;
    int edge_cut;
    double balance_ratio;

    Partition() : edge_cut(0), balance_ratio(0.0) {}

    void print_stats() const;
};

// Task for recursive processing with enhanced metadata
struct DissectionTask
{
    std::vector<int> vertices;
    int level;
    int expected_separator_size;
    double quality_threshold;

    DissectionTask(const std::vector<int> &v, int lvl, int sep_size = 0, double threshold = 0.1);
};