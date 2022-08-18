#pragma once

#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/Common.hpp>
// #include "NonlinearSolver.hpp"
#include <polysolve/LinearSolver.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/ALNLProblem.hpp>
#include <polyfem/solver/SparseNewtonDescentSolver.hpp>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "polyfem/assembler/CUDA_utilities.cuh"
#include "cublas_v2.h"

namespace cppoptlib
{
	template <typename ProblemType>
	bool SparseNewtonDescentSolver<ProblemType>::compute_update_direction_cuda(
			ProblemType &objFunc,
			const Eigen::Matrix<double, -1, 1> &x,
			const Eigen::Matrix<double, -1, 1> &grad,
			Eigen::Matrix<double, -1, 1> &direction)
	{
        return true;
    }
    template class SparseNewtonDescentSolver<polyfem::solver::NLProblem>;
    template class SparseNewtonDescentSolver<polyfem::solver::ALNLProblem>;
}