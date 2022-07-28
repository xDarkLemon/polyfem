#include "NeoHookeanElasticity.hpp"
#include <polyfem/basis/Basis.hpp>
#include <polyfem/autogen/auto_elasticity_rhs.hpp>
#include "cublas_v2.h"
#include <polyfem/utils/MatrixUtils.hpp>
#include <igl/Timer.h>

namespace polyfem
{
	using namespace basis;

	namespace assembler
	{

		template <typename T>
		__device__ T kernel_det(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> &mat, double &result)
		{

			if (mat.rows() == 1)
				result = mat(0);
			else if (mat.rows() == 2)
				result = mat(0, 0) * mat(1, 1) - mat(0, 1) * mat(1, 0);
			else if (mat.rows() == 3)
				result = mat(0, 0) * (mat(1, 1) * mat(2, 2) - mat(1, 2) * mat(2, 1)) - mat(0, 1) * (mat(1, 0) * mat(2, 2) - mat(1, 2) * mat(2, 0)) + mat(0, 2) * (mat(1, 0) * mat(2, 1) - mat(1, 1) * mat(2, 0));
			else
				result = 0;
			return;
		}

		// Compute ∫ ½μ (tr(FᵀF) - 3 - 2ln(J)) + ½λ ln²(J) du
		template <typename T>
		__global__ void compute_energy_gpu_aux(double *displacement,
											   Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> *jac_it_array,
											   Local2Global *global_data,
											   Eigen::Matrix<double, -1, 1, 0, 3, 1> *da,
											   Eigen::Matrix<double, -1, 1, 0, 3, 1> *grad,
											   int N,
											   int bv_N,
											   int gc_N,
											   int n_pts,
											   int _size,
											   double *lambda,
											   double *mu,
											   T *energy_storage)
		{
			int bx = blockIdx.x;
			int tx = threadIdx.x;
			int b_index = bx * NUMBER_THREADS + tx;

			if (b_index < N)
			{
				Eigen::Matrix<double, Eigen::Dynamic, 1> local_dispv(bv_N * _size, 1);
				local_dispv.setZero();
				for (int i = 0; i < bv_N; ++i)
				{
					for (int ii = 0; ii < gc_N; ++ii)
					{
						for (int d = 0; d < _size; ++d)
						{
							//take care of the threads it is not complete
							local_dispv(i * _size + d) += global_data[b_index * bv_N * gc_N + i * gc_N + ii].val * displacement[global_data[b_index * bv_N * gc_N + i * gc_N + ii].index * _size + d];
						}
					}
				}

				T energy = T(0.0);
				Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> def_grad(_size, _size);

				for (long p = 0; p < n_pts; ++p)
				{
					for (long k = 0; k < def_grad.size(); ++k)
						def_grad(k) = T(0);

					for (size_t i = 0; i < bv_N; ++i)
					{
						for (int d = 0; d < _size; ++d)
						{
							for (int c = 0; c < _size; ++c)
							{
								def_grad(d, c) += grad[b_index * bv_N * n_pts + i * n_pts + p](c) * local_dispv(i * _size + d);
							}
						}
					}
					Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> jac_it(_size, _size);
					for (long k = 0; k < jac_it.size(); ++k)
						jac_it(k) = T(jac_it_array[b_index * n_pts + p](k));
					def_grad = def_grad * jac_it;

					//Id + grad d
					for (int d = 0; d < _size; ++d)
						def_grad(d, d) += T(1);

					double _det;
					kernel_det(def_grad, _det);
					const T log_det_j = log(_det);
					const T val = mu[p] / 2 * ((def_grad.transpose() * def_grad).trace() - _size - 2 * log_det_j) + lambda[p] / 2 * log_det_j * log_det_j;

					energy += val * da[b_index](p);
				}

				energy_storage[b_index] = energy;
			}
		}

		void NeoHookeanElasticity::compute_energy_gpu(double *displacement_dev_ptr,
													  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> *jac_it_dev_ptr,
													  Local2Global *global_data_dev_ptr,
													  Eigen::Matrix<double, -1, 1, 0, 3, 1> *da_dev_ptr,
													  Eigen::Matrix<double, -1, 1, 0, 3, 1> *grad_dev_ptr,
													  int n_bases,
													  int bv_N,
													  int gc_N,
													  int n_pts,
													  double *lambda,
													  double *mu,
													  double *energy_storage) const
		{
			int grid = (n_bases % NUMBER_THREADS == 0) ? n_bases / NUMBER_THREADS : n_bases / NUMBER_THREADS + 1;
			int threads = (n_bases > NUMBER_THREADS) ? NUMBER_THREADS : n_bases;
			compute_energy_gpu_aux<double><<<grid, threads>>>(displacement_dev_ptr, jac_it_dev_ptr, global_data_dev_ptr, da_dev_ptr, grad_dev_ptr, n_bases, bv_N, gc_N, n_pts, size(), lambda, mu, energy_storage);
			return;
		}

	} // namespace assembler
} // namespace polyfem
