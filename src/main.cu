#include "../include/fast_nested_dissection.h"
#include "../include/metis_parser.h"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>

void printUsage(const char *program_name)
{
    std::cout << "Usage: " << program_name << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  -metis <filename>    : Process METIS graph file\n";
    std::cout << "  -grid <size>         : Generate and process grid graph of given size\n";
    std::cout << "  -test                : Run small test case\n";
    std::cout << "  -analysis            : Print detailed graph analysis\n";
    std::cout << "  -tree                : Print separator tree structure (warning: can be large)\n";
    std::cout << "  -h, --help           : Show this help message\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << program_name << " -metis graph.txt\n";
    std::cout << "  " << program_name << " -grid 500\n";
    std::cout << "  " << program_name << " -test\n";
}

void generateGridGraph(int grid_size, std::vector<int> &row_ptr,
                       std::vector<int> &col_idx, std::vector<double> &values)
{
    int n = grid_size * grid_size;
    row_ptr.resize(n + 1, 0);
    col_idx.clear();
    values.clear();

    // Build 5-point stencil matrix for 2D grid
    for (int i = 0; i < grid_size; i++)
    {
        for (int j = 0; j < grid_size; j++)
        {
            int idx = i * grid_size + j;

            // Diagonal element (center point)
            col_idx.push_back(idx);
            values.push_back(4.0);

            // Add connections to neighboring points
            if (i > 0)
            { // Up neighbor
                col_idx.push_back((i - 1) * grid_size + j);
                values.push_back(-1.0);
            }
            if (i < grid_size - 1)
            { // Down neighbor
                col_idx.push_back((i + 1) * grid_size + j);
                values.push_back(-1.0);
            }
            if (j > 0)
            { // Left neighbor
                col_idx.push_back(i * grid_size + (j - 1));
                values.push_back(-1.0);
            }
            if (j < grid_size - 1)
            { // Right neighbor
                col_idx.push_back(i * grid_size + (j + 1));
                values.push_back(-1.0);
            }

            row_ptr[idx + 1] = col_idx.size();
        }
    }
}

void processMetisGraph(const std::string &filename, bool show_analysis, bool show_tree)
{
    try
    {
        std::cout << "\n=== PROCESSING METIS GRAPH ===" << std::endl;

        auto start_parse = std::chrono::high_resolution_clock::now();
        MetisGraph graph = MetisGraphParser::parseMetisFile(filename);
        auto end_parse = std::chrono::high_resolution_clock::now();

        std::cout << "Parsing time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end_parse - start_parse).count()
                  << " ms" << std::endl;

        // Create nested dissection instance
        FastNestedDissection nd(graph);

        if (show_analysis)
        {
            nd.printGraphAnalysis();
        }

        // Perform nested dissection
        auto start_nd = std::chrono::high_resolution_clock::now();
        nd.performNestedDissection();
        auto end_nd = std::chrono::high_resolution_clock::now();

        std::cout << "Nested dissection time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end_nd - start_nd).count()
                  << " ms" << std::endl;

        if (show_tree)
        {
            nd.printTreeInfo();
        }

        // Get results
        std::vector<int> permutation = nd.getPermutation();
        std::cout << "Generated permutation vector of size: " << permutation.size() << std::endl;

        // Compute fill reduction estimate
        double fill_reduction = nd.computeFillReduction();
        std::cout << "Estimated fill reduction: " << fill_reduction << "%" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error processing METIS graph: " << e.what() << std::endl;
    }
}

void processGridGraph(int grid_size, bool show_analysis, bool show_tree)
{
    std::cout << "\n=== PROCESSING GRID GRAPH ===" << std::endl;

    int n = grid_size * grid_size;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<double> values;

    auto start_gen = std::chrono::high_resolution_clock::now();
    generateGridGraph(grid_size, row_ptr, col_idx, values);
    auto end_gen = std::chrono::high_resolution_clock::now();

    std::cout << "Grid generation time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_gen - start_gen).count()
              << " ms" << std::endl;

    std::cout << "Created " << n << "x" << n << " matrix with " << col_idx.size() << " nonzeros" << std::endl;

    // Create nested dissection instance
    FastNestedDissection nd(n, grid_size);
    nd.loadMatrix(row_ptr, col_idx, values);

    if (show_analysis)
    {
        nd.printGraphAnalysis();
    }

    // Perform nested dissection
    auto start_nd = std::chrono::high_resolution_clock::now();
    nd.performNestedDissection();
    auto end_nd = std::chrono::high_resolution_clock::now();

    std::cout << "Nested dissection time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_nd - start_nd).count()
              << " ms" << std::endl;

    if (show_tree)
    {
        nd.printTreeInfo();
    }

    // Get results
    std::vector<int> permutation = nd.getPermutation();
    std::cout << "Generated permutation vector of size: " << permutation.size() << std::endl;

    double fill_reduction = nd.computeFillReduction();
    std::cout << "Estimated fill reduction: " << fill_reduction << "%" << std::endl;
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        printUsage(argv[0]);
        return 1;
    }

    bool show_analysis = false;
    bool show_tree = false;
    std::string metis_file;
    int grid_size = 0;
    bool run_test = false;

    // Parse command line arguments
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help")
        {
            printUsage(argv[0]);
            return 0;
        }
        else if (arg == "-metis" && i + 1 < argc)
        {
            metis_file = argv[++i];
        }
        else if (arg == "-grid" && i + 1 < argc)
        {
            grid_size = std::stoi(argv[++i]);
        }
        else if (arg == "-test")
        {
            run_test = true;
        }
        else if (arg == "-analysis")
        {
            show_analysis = true;
        }
        else if (arg == "-tree")
        {
            show_tree = true;
        }
        else
        {
            std::cerr << "Unknown option: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    try
    {
        if (!metis_file.empty())
        {
            processMetisGraph(metis_file, show_analysis, show_tree);
        }
        else if (grid_size > 0)
        {
            processGridGraph(grid_size, show_analysis, show_tree);
        }
        else if (run_test)
        {
            std::cout << "Running small test case..." << std::endl;
            processGridGraph(50, true, false); // Small test grid
        }
        else
        {
            std::cerr << "No valid operation specified. Use -h for help." << std::endl;
            return 1;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}