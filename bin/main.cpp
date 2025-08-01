#include "../include/nested_dissection.h"
#include "../include/mtx_reader.h"
#include <iostream>
#include <chrono>
#include <string>
#include <cmath>
#include <exception>

void print_usage(const char *program_name)
{
    std::cout << "Usage:" << std::endl;
    std::cout << "  " << program_name << "                    # Default 6x6 grid" << std::endl;
    std::cout << "  " << program_name << " 10                 # 10x10 grid" << std::endl;
    std::cout << "  " << program_name << " matrix.mtx         # MTX file" << std::endl;
}

void print_grid_layout(int grid_size, int max_display = 10)
{
    std::cout << "Grid layout (" << grid_size << "x" << grid_size << "):" << std::endl;
    for (int i = 0; i < grid_size && i < max_display; i++)
    {
        for (int j = 0; j < grid_size && j < max_display; j++)
        {
            int vertex = i * grid_size + j;
            printf("%3d ", vertex);
        }
        if (grid_size > max_display)
            std::cout << "...";
        std::cout << std::endl;
    }
    if (grid_size > max_display)
        std::cout << "..." << std::endl;
    std::cout << std::endl;
}

void print_ordering_results(const std::vector<int> &ordering)
{
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
}

Graph load_graph_from_args(int argc, char *argv[], std::string &graph_type)
{
    Graph graph(0, 0);
    graph_type = "grid";

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
                graph = ImprovedGPUNestedDissection::create_test_grid_graph();
                graph_type = "grid";
            }
        }
        else
        {
            try
            {
                int grid_size = std::stoi(filename);
                if (grid_size <= 0 || grid_size > 1000)
                {
                    throw std::invalid_argument("Grid size must be between 1 and 1000");
                }
                std::cout << "Creating " << grid_size << "x" << grid_size << " test grid..." << std::endl;
                graph = ImprovedGPUNestedDissection::create_test_grid_graph(grid_size);
            }
            catch (const std::exception &e)
            {
                std::cerr << "Error parsing grid size: " << e.what() << std::endl;
                std::cout << "Falling back to default 6x6 grid..." << std::endl;
                graph = ImprovedGPUNestedDissection::create_test_grid_graph();
            }
        }
    }
    else
    {
        std::cout << "Creating default 6x6 test grid..." << std::endl;
        graph = ImprovedGPUNestedDissection::create_test_grid_graph();
    }

    return graph;
}

int main(int argc, char *argv[])
{
    std::cout << "=== GPU Nested Dissection with MTX Support ===" << std::endl;

    try
    {
        // Load graph based on command line arguments
        std::string graph_type;
        Graph graph = load_graph_from_args(argc, argv, graph_type);

        std::cout << "Graph loaded: " << graph.n_vertices << " vertices, "
                  << graph.n_edges << " edges" << std::endl;

        // Print grid layout for grid graphs
        if (graph_type == "grid")
        {
            int grid_size = static_cast<int>(std::sqrt(graph.n_vertices));
            print_grid_layout(grid_size);
        }

        // Initialize nested dissection algorithm
        ImprovedGPUNestedDissection nd(graph.n_vertices);

        std::cout << "Computing improved nested dissection ordering..." << std::endl;

        // Time the computation
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<int> ordering = nd.compute_ordering(graph);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        // Print results
        print_ordering_results(ordering);
        std::cout << "\nComputation time: " << duration.count() << " microseconds" << std::endl;

        // Verify ordering completeness
        if (ordering.size() != graph.n_vertices)
        {
            std::cerr << "Warning: Ordering incomplete! Expected " << graph.n_vertices
                      << " vertices, got " << ordering.size() << std::endl;
        }
        else
        {
            std::cout << "Ordering verification: Complete ✓" << std::endl;
        }

        print_usage(argv[0]);
        std::cout << "Done!" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    catch (...)
    {
        std::cerr << "Unknown fatal error occurred" << std::endl;
        return 1;
    }

    return 0;
}