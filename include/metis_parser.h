#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>

// Structure to hold METIS graph data
struct MetisGraph
{
    int n;    // Number of vertices
    int m;    // Number of edges
    int fmt;  // Format parameter (3-bit binary)
    int ncon; // Number of vertex weights per vertex

    // Graph structure (CSR format)
    std::vector<int> row_ptr;         // Size n+1
    std::vector<int> col_idx;         // Size 2*m (since each edge appears twice in adjacency list)
    std::vector<double> edge_weights; // Edge weights (if fmt & 1)

    // Vertex data
    std::vector<int> vertex_sizes;                   // Vertex sizes (if fmt & 4)
    std::vector<std::vector<double>> vertex_weights; // Vertex weights (if fmt & 2)

    // Parsed format flags
    bool has_edge_weights;
    bool has_vertex_weights;
    bool has_vertex_sizes;

    MetisGraph() : n(0), m(0), fmt(0), ncon(1),
                   has_edge_weights(false), has_vertex_weights(false), has_vertex_sizes(false) {}
};

class MetisGraphParser
{
public:
    static MetisGraph parseMetisFile(const std::string &filename);
    static void printGraphInfo(const MetisGraph &graph);
    static void validateGraph(const MetisGraph &graph);

private:
    static void parseHeaderLine(const std::string &line, MetisGraph &graph);
    static void parseVertexLine(const std::string &line, int vertex_id, MetisGraph &graph);
    static void parseFmtParameter(int fmt, MetisGraph &graph);
    static bool isCommentLine(const std::string &line);
    static std::vector<std::string> tokenize(const std::string &line);
};