#pragma once

#include <Eigen/Sparse>
#include <Eigen/Core>
#include <vector>


using namespace Eigen;
using namespace std;

// global variables
extern Eigen::MatrixXd test_vertices;
extern Eigen::MatrixXd init_vertices;
extern std::vector<int> test_boundary_nodes;

inline Eigen::MatrixXd remove_boundary_vertices(const Eigen::MatrixXd &vertices, const std::vector<int> &boundary_nodes)
{
    // Remove boundary vertices
    if (boundary_nodes.empty())
    {
        return vertices;
    }
    else
    {
        std::vector<int> order_nodes = boundary_nodes;
        std::sort(order_nodes.begin(), order_nodes.end());
        Eigen::MatrixXd out_vertices;
        std::vector<int> keep;
        for (int i = 0; i < vertices.rows(); i++)
        {
            if (!std::binary_search(order_nodes.begin(), order_nodes.end(),i))
            {
                keep.push_back(i);
            }
        }
        out_vertices = vertices(keep, Eigen::all);
        return out_vertices;
    }
}