#include "../include/fast_nested_dissection.h"
#include <iostream>
#include <vector>

int main()
{
#ifdef SMALL_TEST_MATRIX
    int grid_size = 100; // Smaller test matrix for quick testing
#else
    int grid_size = 1000; // Full size matrix
#endif
    int n = grid_size * grid_size;

    std::vector<int> row_ptr(n + 1, 0);
    std::vector<int> col_idx;
    std::vector<double> values;

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
            if (i > 0) // Up neighbor
            {
                col_idx.push_back((i - 1) * grid_size + j);
                values.push_back(-1.0);
            }
            if (i < grid_size - 1) // Down neighbor
            {
                col_idx.push_back((i + 1) * grid_size + j);
                values.push_back(-1.0);
            }
            if (j > 0) // Left neighbor
            {
                col_idx.push_back(i * grid_size + (j - 1));
                values.push_back(-1.0);
            }
            if (j < grid_size - 1) // Right neighbor
            {
                col_idx.push_back(i * grid_size + (j + 1));
                values.push_back(-1.0);
            }

            row_ptr[idx + 1] = col_idx.size();
        }
    }

    std::cout << "Created " << n << "x" << n << " matrix with " << col_idx.size() << " nonzeros" << std::endl;

    // Test the fast nested dissection algorithm
    std::cout << "\n=== SPEED-OPTIMIZED NESTED DISSECTION ===" << std::endl;

    FastNestedDissection fast_nd(n, grid_size);
    fast_nd.loadMatrix(row_ptr, col_idx, values);
    fast_nd.performNestedDissection();

    // Uncomment to see tree structure (warning: can be very large output)
    // fast_nd.printTreeInfo();

    // Get the permutation vector
    std::vector<int> permutation = fast_nd.getPermutation();
    std::cout << "Generated permutation vector of size: " << permutation.size() << std::endl;

    return 0;
}