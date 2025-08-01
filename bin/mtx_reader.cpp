#include "../include/mtx_reader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <memory>

MTXReader::MTXGraph MTXReader::read_mtx_file(const std::string &filename)
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

Graph MTXReader::convert_to_csr_graph(const MTXGraph &mtx_graph)
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
    cleanup_adjacency_list(adj_list);

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

void MTXReader::cleanup_adjacency_list(std::vector<std::vector<std::pair<int, int>>> &adj_list)
{
    for (auto &neighbors : adj_list)
    {
        // Sort by neighbor vertex ID
        std::sort(neighbors.begin(), neighbors.end());

        // Remove duplicates, keeping the one with maximum weight
        auto it = neighbors.begin();
        while (it != neighbors.end())
        {
            auto next_it = it + 1;
            while (next_it != neighbors.end() && next_it->first == it->first)
            {
                // Keep the edge with maximum weight
                if (next_it->second > it->second)
                {
                    it->second = next_it->second;
                }
                next_it = neighbors.erase(next_it);
            }
            ++it;
        }
    }
}