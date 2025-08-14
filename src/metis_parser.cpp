#include "metis_parser.h"

MetisGraph MetisGraphParser::parseMetisFile(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    MetisGraph graph;
    std::string line;
    bool header_parsed = false;
    int current_vertex = 0;

    std::cout << "Parsing METIS file: " << filename << std::endl;

    while (std::getline(file, line))
    {
        // Skip comment lines
        if (isCommentLine(line) || line.empty())
        {
            continue;
        }

        if (!header_parsed)
        {
            parseHeaderLine(line, graph);
            header_parsed = true;

            // Initialize data structures based on header
            graph.row_ptr.resize(graph.n + 1, 0);
            graph.col_idx.reserve(2 * graph.m); // Estimate

            if (graph.has_edge_weights)
            {
                graph.edge_weights.reserve(2 * graph.m);
            }
            if (graph.has_vertex_sizes)
            {
                graph.vertex_sizes.resize(graph.n);
            }
            if (graph.has_vertex_weights)
            {
                graph.vertex_weights.resize(graph.n, std::vector<double>(graph.ncon));
            }

            std::cout << "Header parsed - n=" << graph.n << ", m=" << graph.m
                      << ", fmt=" << graph.fmt << ", ncon=" << graph.ncon << std::endl;
        }
        else
        {
            if (current_vertex >= graph.n)
            {
                throw std::runtime_error("Too many vertex lines in file");
            }
            parseVertexLine(line, current_vertex, graph);
            current_vertex++;
        }
    }

    if (current_vertex != graph.n)
    {
        throw std::runtime_error("Number of vertex lines doesn't match header");
    }

    // Finalize row_ptr (convert from edge counts to cumulative offsets)
    int total_edges = 0;
    for (int i = 0; i <= graph.n; i++)
    {
        int temp = graph.row_ptr[i];
        graph.row_ptr[i] = total_edges;
        total_edges += temp;
    }

    validateGraph(graph);
    printGraphInfo(graph);

    return graph;
}

void MetisGraphParser::parseHeaderLine(const std::string &line, MetisGraph &graph)
{
    std::vector<std::string> tokens = tokenize(line);

    if (tokens.size() < 2 || tokens.size() > 4)
    {
        throw std::runtime_error("Invalid header line format");
    }

    graph.n = std::stoi(tokens[0]);
    graph.m = std::stoi(tokens[1]);

    if (tokens.size() >= 3)
    {
        graph.fmt = std::stoi(tokens[2]);
    }

    if (tokens.size() == 4)
    {
        graph.ncon = std::stoi(tokens[3]);
    }

    parseFmtParameter(graph.fmt, graph);
}

void MetisGraphParser::parseVertexLine(const std::string &line, int vertex_id, MetisGraph &graph)
{
    std::vector<std::string> tokens = tokenize(line);
    int token_idx = 0;

    // Parse vertex size if present
    if (graph.has_vertex_sizes)
    {
        if (token_idx >= tokens.size())
        {
            throw std::runtime_error("Missing vertex size on line for vertex " + std::to_string(vertex_id + 1));
        }
        graph.vertex_sizes[vertex_id] = std::stoi(tokens[token_idx++]);
    }

    // Parse vertex weights if present
    if (graph.has_vertex_weights)
    {
        for (int i = 0; i < graph.ncon; i++)
        {
            if (token_idx >= tokens.size())
            {
                throw std::runtime_error("Missing vertex weight on line for vertex " + std::to_string(vertex_id + 1));
            }
            graph.vertex_weights[vertex_id][i] = std::stod(tokens[token_idx++]);
        }
    }

    // Parse adjacency list
    int edge_count = 0;
    while (token_idx < tokens.size())
    {
        // Parse neighbor vertex (convert from 1-based to 0-based indexing)
        int neighbor = std::stoi(tokens[token_idx++]) - 1;
        if (neighbor < 0 || neighbor >= graph.n)
        {
            throw std::runtime_error("Invalid neighbor vertex index: " + std::to_string(neighbor + 1));
        }

        graph.col_idx.push_back(neighbor);
        edge_count++;

        // Parse edge weight if present
        if (graph.has_edge_weights)
        {
            if (token_idx >= tokens.size())
            {
                throw std::runtime_error("Missing edge weight for vertex " + std::to_string(vertex_id + 1));
            }
            double weight = std::stod(tokens[token_idx++]);
            if (weight <= 0)
            {
                throw std::runtime_error("Edge weight must be positive");
            }
            graph.edge_weights.push_back(weight);
        }
    }

    // Store edge count in row_ptr (will be converted to cumulative later)
    graph.row_ptr[vertex_id + 1] = edge_count;
}

void MetisGraphParser::parseFmtParameter(int fmt, MetisGraph &graph)
{
    // fmt is a 3-bit binary number
    graph.has_edge_weights = (fmt & 1) != 0;   // 1st bit (rightmost)
    graph.has_vertex_weights = (fmt & 2) != 0; // 2nd bit
    graph.has_vertex_sizes = (fmt & 4) != 0;   // 3rd bit
}

bool MetisGraphParser::isCommentLine(const std::string &line)
{
    return !line.empty() && line[0] == '%';
}

std::vector<std::string> MetisGraphParser::tokenize(const std::string &line)
{
    std::vector<std::string> tokens;
    std::istringstream iss(line);
    std::string token;

    while (iss >> token)
    {
        tokens.push_back(token);
    }

    return tokens;
}

void MetisGraphParser::printGraphInfo(const MetisGraph &graph)
{
    std::cout << "\n=== METIS Graph Information ===" << std::endl;
    std::cout << "Vertices: " << graph.n << std::endl;
    std::cout << "Edges: " << graph.m << std::endl;
    std::cout << "Format: " << graph.fmt << std::endl;
    std::cout << "Vertex constraints: " << graph.ncon << std::endl;
    std::cout << "Has edge weights: " << (graph.has_edge_weights ? "Yes" : "No") << std::endl;
    std::cout << "Has vertex weights: " << (graph.has_vertex_weights ? "Yes" : "No") << std::endl;
    std::cout << "Has vertex sizes: " << (graph.has_vertex_sizes ? "Yes" : "No") << std::endl;
    std::cout << "Total adjacency entries: " << graph.col_idx.size() << std::endl;

    // Check if graph is roughly symmetric (undirected)
    int total_edges = graph.col_idx.size();
    std::cout << "Expected edges for undirected graph: " << 2 * graph.m << std::endl;
    std::cout << "Actual adjacency entries: " << total_edges << std::endl;

    if (std::abs(total_edges - 2 * graph.m) > graph.m * 0.1)
    {
        std::cout << "Warning: Edge count mismatch - graph may not be properly undirected" << std::endl;
    }
}

void MetisGraphParser::validateGraph(const MetisGraph &graph)
{
    // Basic validation
    if (graph.n <= 0 || graph.m < 0)
    {
        throw std::runtime_error("Invalid graph dimensions");
    }

    if (graph.row_ptr.size() != graph.n + 1)
    {
        throw std::runtime_error("Invalid row_ptr size");
    }

    // Check CSR structure
    for (int i = 0; i < graph.n; i++)
    {
        if (graph.row_ptr[i] > graph.row_ptr[i + 1])
        {
            throw std::runtime_error("Invalid CSR structure - non-monotonic row_ptr");
        }
    }

    // Validate adjacency indices
    for (int idx : graph.col_idx)
    {
        if (idx < 0 || idx >= graph.n)
        {
            throw std::runtime_error("Invalid adjacency index: " + std::to_string(idx));
        }
    }

    // Validate dimensions match expected sizes
    if (graph.has_edge_weights && graph.edge_weights.size() != graph.col_idx.size())
    {
        throw std::runtime_error("Edge weights size mismatch");
    }

    if (graph.has_vertex_sizes && graph.vertex_sizes.size() != graph.n)
    {
        throw std::runtime_error("Vertex sizes size mismatch");
    }

    if (graph.has_vertex_weights && graph.vertex_weights.size() != graph.n)
    {
        throw std::runtime_error("Vertex weights size mismatch");
    }

    std::cout << "Graph validation passed!" << std::endl;
}