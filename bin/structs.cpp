#include "../include/structs.h"
#include <cuda_runtime.h>
#include <iostream>
#include <utility>

// Graph implementation
Graph::Graph(int n_v, int n_e) : n_vertices(n_v), n_edges(n_e)
{
    cudaMalloc(&row_ptr, (n_vertices + 1) * sizeof(int));
    cudaMalloc(&col_idx, n_edges * sizeof(int));
    cudaMalloc(&edge_weights, n_edges * sizeof(int));
    cudaMalloc(&vertex_weights, n_vertices * sizeof(int));
}

Graph::~Graph()
{
    if (row_ptr)
        cudaFree(row_ptr);
    if (col_idx)
        cudaFree(col_idx);
    if (edge_weights)
        cudaFree(edge_weights);
    if (vertex_weights)
        cudaFree(vertex_weights);
}

Graph::Graph(Graph &&other) noexcept
    : n_vertices(other.n_vertices), n_edges(other.n_edges), row_ptr(other.row_ptr), col_idx(other.col_idx), edge_weights(other.edge_weights), vertex_weights(other.vertex_weights)
{
    // Nullify the source object's pointers
    other.row_ptr = nullptr;
    other.col_idx = nullptr;
    other.edge_weights = nullptr;
    other.vertex_weights = nullptr;
    other.n_vertices = 0;
    other.n_edges = 0;
}

Graph &Graph::operator=(Graph &&other) noexcept
{
    if (this != &other)
    {
        // Free existing resources
        if (row_ptr)
            cudaFree(row_ptr);
        if (col_idx)
            cudaFree(col_idx);
        if (edge_weights)
            cudaFree(edge_weights);
        if (vertex_weights)
            cudaFree(vertex_weights);

        // Move data from other
        n_vertices = other.n_vertices;
        n_edges = other.n_edges;
        row_ptr = other.row_ptr;
        col_idx = other.col_idx;
        edge_weights = other.edge_weights;
        vertex_weights = other.vertex_weights;

        // Nullify the source object's pointers
        other.row_ptr = nullptr;
        other.col_idx = nullptr;
        other.edge_weights = nullptr;
        other.vertex_weights = nullptr;
        other.n_vertices = 0;
        other.n_edges = 0;
    }
    return *this;
}

// Partition implementation
void Partition::print_stats() const
{
    std::cout << "    Partition stats - Left: " << left_vertices.size()
              << ", Right: " << right_vertices.size()
              << ", Separator: " << separator_vertices.size()
              << ", Edge cut: " << edge_cut
              << ", Balance: " << std::fixed << std::setprecision(3) << balance_ratio << std::endl;
}

// DissectionTask implementation
DissectionTask::DissectionTask(const std::vector<int> &v, int lvl, int sep_size, double threshold)
    : vertices(v), level(lvl), expected_separator_size(sep_size), quality_threshold(threshold)
{
}