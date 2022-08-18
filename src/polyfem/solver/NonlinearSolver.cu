#pragma once

#include <polyfem/Common.hpp>

#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/Timer.hpp>

// Line search methods
// #include "line_search/LineSearch.hpp"

#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/NonlinearSolver.hpp>

#include <cppoptlib/solver/isolver.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/inner_product.h>

#include <iostream>

namespace cppoptlib
{
    // typedef Eigen::Matrix<double, -1, 1> TVector;

    template <typename ProblemType /*, int Ord*/>
    void NonlinearSolver<ProblemType>::minimize_gpu(ProblemType &objFunc, Eigen::Matrix<double, -1, 1> &x)
    {
    }
    template class NonlinearSolver<polyfem::solver::NLProblem>;
}