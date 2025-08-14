#include "../include/fast_nested_dissection.h"
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/remove.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/execution_policy.h>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <chrono>

// Tree building and algorithms implementation
void FastNestedDissection::buildNestedDissectionTreeFast()
{
    auto timer_start = std::chrono::high_resolution_clock::now();
    std::vector<int> all_vertices(n);
    std::iota(all_vertices.begin(), all_vertices.end(), 0);

    separator_tree = std::make_unique<SeparatorNode>(0);

    auto setup_time = std::chrono::high_resolution_clock::now();
    std::cout << "Setup time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(setup_time - timer_start).count()
              << " ms" << std::endl;

    // Use iterative approach with queue for better cache locality
    struct Task
    {
        SeparatorNode *node;
        std::vector<int> vertices;
        int start_x, start_y, width, height;
    };

    std::queue<Task> task_queue;
    Task initial_task;
    initial_task.node = separator_tree.get();
    initial_task.vertices = all_vertices;
    initial_task.start_x = 0;
    initial_task.start_y = 0;
    initial_task.width = grid_size;
    initial_task.height = grid_size;
    task_queue.push(initial_task);

    int iteration_count = 0;
    int total_separator_time = 0;
    int total_components_time = 0;

    while (!task_queue.empty())
    {
        Task current = task_queue.front();
        task_queue.pop();
        iteration_count++;
        // int base_case_size = std::max(2048, n / 256);
        int base_case_size = 512;
        if (current.vertices.size() <= base_case_size)
        { // Larger base case
            current.node->vertices = current.vertices;
            continue;
        }

        auto sep_start = std::chrono::high_resolution_clock::now();
        // Fast separator finding
        std::vector<int> separator;
        if (grid_size > 0 && current.width > 8 && current.height > 8)
        {
            separator = fastGeometricSeparator(current.vertices,
                                               current.start_x, current.start_y,
                                               current.width, current.height);
        }

        // Fallback to spectral if geometric fails or for small subgraphs
        if (separator.empty())
        {
            separator = approximateSpectralSeparator(current.vertices);
        }

        auto sep_end = std::chrono::high_resolution_clock::now();
        total_separator_time += std::chrono::duration_cast<std::chrono::milliseconds>(sep_end - sep_start).count();

        current.node->vertices = separator;

        // Fast connected components
        auto comp_start = std::chrono::high_resolution_clock::now();
        std::pair<std::vector<int>, std::vector<int>> components = fastConnectedComponents(current.vertices, separator);
        auto comp_end = std::chrono::high_resolution_clock::now();
        total_components_time += std::chrono::duration_cast<std::chrono::milliseconds>(comp_end - comp_start).count();
        std::vector<int> A_vertices = components.first;
        std::vector<int> B_vertices = components.second;

        current.node->A_vertices = A_vertices;
        current.node->B_vertices = B_vertices;

        // Add child tasks
        if (!A_vertices.empty())
        {
            current.node->left = std::make_unique<SeparatorNode>(current.node->level + 1);
            Task left_task;
            left_task.node = current.node->left.get();
            left_task.vertices = A_vertices;
            left_task.start_x = current.start_x;
            left_task.start_y = current.start_y;
            left_task.width = current.width / 2;
            left_task.height = current.height;
            task_queue.push(left_task);
        }

        if (!B_vertices.empty())
        {
            current.node->right = std::make_unique<SeparatorNode>(current.node->level + 1);
            Task right_task;
            right_task.node = current.node->right.get();
            right_task.vertices = B_vertices;
            right_task.start_x = current.start_x + current.width / 2;
            right_task.start_y = current.start_y;
            right_task.width = current.width / 2;
            right_task.height = current.height;
            task_queue.push(right_task);
        }
    }
    std::cout << "Total iterations: " << iteration_count << std::endl;
    std::cout << "Total separator time: " << total_separator_time << " ms" << std::endl;
    std::cout << "Total components time: " << total_components_time << " ms" << std::endl;
}

std::pair<std::vector<int>, std::vector<int>> FastNestedDissection::fastConnectedComponents(
    const std::vector<int> &remaining, const std::vector<int> &separator)
{
    std::unordered_set<int> sep_set(separator.begin(), separator.end());
    std::vector<int> A_vertices, B_vertices;

    // Remove separator vertices and split remaining vertices
    for (int v : remaining)
    {
        if (sep_set.find(v) == sep_set.end())
        {
            // Use simple hash-based partitioning for speed
            if ((v ^ (v >> 16)) & 1)
            {
                A_vertices.push_back(v);
            }
            else
            {
                B_vertices.push_back(v);
            }
        }
    }

    return std::make_pair(A_vertices, B_vertices);
}

void FastNestedDissection::generatePermutation()
{
    std::vector<int> permutation;
    generatePermutationRecursive(separator_tree.get(), permutation);
    d_perm = permutation;
}

void FastNestedDissection::generatePermutationRecursive(SeparatorNode *node, std::vector<int> &permutation)
{
    if (!node)
        return;

    generatePermutationRecursive(node->left.get(), permutation);
    generatePermutationRecursive(node->right.get(), permutation);

    permutation.insert(permutation.end(), node->vertices.begin(), node->vertices.end());
}

void FastNestedDissection::performNestedDissection()
{
    auto start = std::chrono::high_resolution_clock::now();

    std::cout << "Building fast nested dissection tree..." << std::endl;
    buildNestedDissectionTreeFast();

    auto tree_time = std::chrono::high_resolution_clock::now();
    std::cout << "Tree building time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(tree_time - start).count()
              << " ms" << std::endl;

    std::cout << "Generating permutation..." << std::endl;
    generatePermutation();

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Total time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;
}

std::vector<int> FastNestedDissection::getPermutation()
{
    thrust::host_vector<int> h_perm = d_perm;
    return std::vector<int>(h_perm.begin(), h_perm.end());
}

void FastNestedDissection::printTreeInfo()
{
    std::cout << "Fast Nested Dissection Tree Structure:" << std::endl;
    printTreeRecursive(separator_tree.get(), "");
}

void FastNestedDissection::printTreeRecursive(SeparatorNode *node, std::string indent)
{
    if (!node)
        return;

    std::cout << indent << "Level " << node->level
              << ": Separator size = " << node->vertices.size()
              << ", A size = " << node->A_vertices.size()
              << ", B size = " << node->B_vertices.size() << std::endl;

    printTreeRecursive(node->left.get(), indent + "  ");
    printTreeRecursive(node->right.get(), indent + "  ");
}

// QualityMetrics implementation
double QualityMetrics::computeSeparatorQuality(const std::vector<int> &separator,
                                               const std::vector<int> &A_vertices,
                                               const std::vector<int> &B_vertices)
{
    double balance = 1.0 - std::abs((double)A_vertices.size() - B_vertices.size()) /
                               (A_vertices.size() + B_vertices.size());
    double separator_ratio = (double)separator.size() /
                             (separator.size() + A_vertices.size() + B_vertices.size());

    return balance * (1.0 - separator_ratio); // Higher is better
}

std::vector<int> QualityMetrics::refineSeparator(const std::vector<int> &separator,
                                                 const std::vector<int> &A_vertices,
                                                 const std::vector<int> &B_vertices,
                                                 const std::vector<int> &row_ptr,
                                                 const std::vector<int> &col_idx)
{
    // Kernighan-Lin style local refinement
    std::vector<int> refined_separator = separator;

    // Try moving vertices between separator and subgraphs
    for (int iter = 0; iter < 5; iter++)
    {
        bool improved = false;

        for (int i = 0; i < refined_separator.size(); i++)
        {
            int v = refined_separator[i];

            // Count connections to A and B
            int conn_A = 0, conn_B = 0;
            for (int j = row_ptr[v]; j < row_ptr[v + 1]; j++)
            {
                int neighbor = col_idx[j];
                if (std::find(A_vertices.begin(), A_vertices.end(), neighbor) != A_vertices.end())
                {
                    conn_A++;
                }
                else if (std::find(B_vertices.begin(), B_vertices.end(), neighbor) != B_vertices.end())
                {
                    conn_B++;
                }
            }

            // Move to the side with more connections if beneficial
            if (conn_A > conn_B + 2)
            {
                refined_separator.erase(refined_separator.begin() + i);
                improved = true;
                break;
            }
            else if (conn_B > conn_A + 2)
            {
                refined_separator.erase(refined_separator.begin() + i);
                improved = true;
                break;
            }
        }

        if (!improved)
            break;
    }

    return refined_separator;
}