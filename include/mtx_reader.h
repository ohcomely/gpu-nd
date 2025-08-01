#pragma once

#include "structs.h"
#include <string>
#include <vector>

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

        MTXGraph() : n_vertices(0), n_edges(0), is_symmetric(false), is_pattern(false) {}
    };

    /**
     * Read an MTX file and return the graph data
     * @param filename Path to the MTX file
     * @return MTXGraph structure containing the parsed data
     * @throws std::runtime_error if file cannot be read or is invalid
     */
    static MTXGraph read_mtx_file(const std::string &filename);

    /**
     * Convert MTX graph data to CSR format on GPU
     * @param mtx_graph The MTX graph data to convert
     * @return Graph object in CSR format ready for GPU processing
     */
    static Graph convert_to_csr_graph(const MTXGraph &mtx_graph);

private:
    /**
     * Remove duplicate edges and sort adjacency lists
     * @param adj_list Adjacency list to clean up
     */
    static void cleanup_adjacency_list(std::vector<std::vector<std::pair<int, int>>> &adj_list);
};