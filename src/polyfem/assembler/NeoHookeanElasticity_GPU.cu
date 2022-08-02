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

		template <int dim>
		__device__ Eigen::Matrix<double, dim, 1> kernel_cross(const Eigen::Matrix<double, dim, 1> &x, const Eigen::Matrix<double, dim, 1> &y)
		{

			Eigen::Matrix<double, dim, 1> z;
			z.setZero();

			z(0) = x(1) * y(2) - x(2) * y(1);
			z(1) = x(2) * y(0) - x(0) * y(2);
			z(2) = x(0) * y(1) - x(1) * y(0);

			return z;
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
											   T *energy_val)
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
							//threads allocation jumps the size of the respective vectors
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

				typedef cub::BlockReduce<double, 32> BlockReduce;
				// Allocate shared memory for BlockReduce
				__shared__ typename BlockReduce::TempStorage temp_storage;

				double result_ = BlockReduce(temp_storage).Sum(energy);
				if (tx == 0)
				{
					atomicAdd(&energy_val[0], result_);
				}
			}
		}

		template <int n_basis, int dim>
		__global__ void compute_energy_aux_gradient_fast_GPU(int N_basis_global,
															 double *displacement,
															 Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> *jac_it_array,
															 Local2Global *global_data,
															 Eigen::Matrix<double, -1, 1, 0, 3, 1> *da,
															 Eigen::Matrix<double, -1, -1, 0, 3, 3> *grad_v,
															 int N,
															 int bv_N,
															 int gc_N,
															 int n_pts,
															 int size_,
															 double *lambda,
															 double *mu,
															 double *result_vec)
		{
			Eigen::Matrix<double, -1, -1> vec(N_basis_global * size_, 1);
			vec.setZero();

			extern __shared__ double shared_vec[];

			int bx = blockIdx.x;
			int tx = threadIdx.x;
			int b_index = bx * NUMBER_THREADS + tx;

			if (b_index < N)
			{
				Eigen::Matrix<double, n_basis, dim> local_disp(bv_N, size_);
				local_disp.setZero();
				for (int i = 0; i < bv_N; ++i)
				{
					for (int ii = 0; ii < gc_N; ++ii)
					{
						for (int d = 0; d < size_; ++d)
						{
							//threads allocation jumps the size of the respective vectors
							local_disp(i, d) += global_data[b_index * bv_N * gc_N + i * gc_N + ii].val * displacement[global_data[b_index * bv_N * gc_N + i * gc_N + ii].index * size_ + d];
						}
					}
				}
				Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, dim, dim> def_grad(size_, size_);

				Eigen::Matrix<double, n_basis, dim> G(bv_N, size_);
				G.setZero();

				for (long p = 0; p < n_pts; ++p)
				{
					Eigen::Matrix<double, n_basis, dim> grad(bv_N, size_);
					for (size_t i = 0; i < bv_N; ++i)
					{
						grad.row(i) = grad_v[b_index * bv_N * n_pts + i * n_pts].row(p);
					}
					Eigen::Matrix<double, dim, dim> jac_it;
					for (long k = 0; k < jac_it.size(); ++k)
						jac_it(k) = jac_it_array[b_index * n_pts + p](k);

					//Id + grad d
					def_grad = local_disp.transpose() * grad * jac_it + Eigen::Matrix<double, dim, dim>::Identity(size_, size_);

					double J;
					kernel_det(def_grad, J);
					const double log_det_j = log(J);

					Eigen::Matrix<double, dim, dim> delJ_delF(size_, size_);
					delJ_delF.setZero();

					if (dim == 2)
					{

						delJ_delF(0, 0) = def_grad(1, 1);
						delJ_delF(0, 1) = -def_grad(1, 0);
						delJ_delF(1, 0) = -def_grad(0, 1);
						delJ_delF(1, 1) = def_grad(0, 0);
					}

					else if (dim == 3)
					{

						Eigen::Matrix<double, dim, 1> u(def_grad.rows());
						Eigen::Matrix<double, dim, 1> v(def_grad.rows());
						Eigen::Matrix<double, dim, 1> w(def_grad.rows());

						u = def_grad.col(0);
						v = def_grad.col(1);
						w = def_grad.col(2);

						delJ_delF.col(0) = kernel_cross<dim>(v, w);
						delJ_delF.col(1) = kernel_cross<dim>(w, u);
						delJ_delF.col(2) = kernel_cross<dim>(u, v);
					}

					Eigen::Matrix<double, n_basis, dim> delF_delU = grad * jac_it;

					Eigen::Matrix<double, dim, dim> gradient_temp = mu[p] * def_grad - mu[p] * (1 / J) * delJ_delF + lambda[p] * log_det_j * (1 / J) * delJ_delF;
					Eigen::Matrix<double, n_basis, dim> gradient = delF_delU * gradient_temp.transpose();

					double val = mu[p] / 2 * ((def_grad.transpose() * def_grad).trace() - size_ - 2 * log_det_j) + lambda[p] / 2 * log_det_j * log_det_j;

					G.noalias() += gradient * da[b_index](p);
				}

				Eigen::Matrix<double, dim, n_basis> G_T = G.transpose();

				constexpr int N = (n_basis == Eigen::Dynamic) ? Eigen::Dynamic : n_basis * dim;
				Eigen::Matrix<double, N, 1> temp(Eigen::Map<Eigen::Matrix<double, N, 1>>(G_T.data(), G_T.size()));

				for (int j = 0; j < bv_N; ++j)
				{
					for (int m = 0; m < size_; ++m)
					{
						const double local_value = temp(j * size_ + m);
						if (std::abs(local_value) < 1e-30)
						{
							continue;
						}

						for (size_t jj = 0; jj < gc_N; ++jj)
						{
							const auto gj = global_data[b_index * bv_N * gc_N + j * gc_N + jj].index * size_ + m;
							const auto wj = global_data[b_index * bv_N * gc_N + j * gc_N + jj].val;

							vec(gj) += local_value * wj;
						}
					}
				}

				typedef cub::BlockReduce<double, 32> BlockReduce;
				// Allocate shared memory for BlockReduce
				__shared__ typename BlockReduce::TempStorage temp_storage;

				for (int i = 0; i < N_basis_global * size_; i++)
				{
					double result_ = BlockReduce(temp_storage).Sum(vec(i));
					if (tx == 0)
					{
						shared_vec[i] = result_;
						atomicAdd(&result_vec[i], shared_vec[i]);
					}
				}

				//				for (int i = 0; i < N_basis_global * size_; i++)
				//				{
				//					if (tx == 0)
				//					{
				//						//	atomicAdd(&result_vec[i], shared_vec[i]);
				//					}
				//				}
			}
		}

		int NeoHookeanElasticity::compute_energy_gpu(double *displacement_dev_ptr,
													 Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> *jac_it_dev_ptr,
													 Local2Global *global_data_dev_ptr,
													 Eigen::Matrix<double, -1, 1, 0, 3, 1> *da_dev_ptr,
													 Eigen::Matrix<double, -1, 1, 0, 3, 1> *grad_dev_ptr,
													 int n_bases,
													 int bv_N,
													 int gc_N,
													 int n_pts,
													 double *lambda,
													 double *mu) const
		//													 double *energy_storage) const
		{
			int grid = (n_bases % NUMBER_THREADS == 0) ? n_bases / NUMBER_THREADS : n_bases / NUMBER_THREADS + 1;
			int threads = (n_bases > NUMBER_THREADS) ? NUMBER_THREADS : n_bases;

			thrust::device_vector<double> energy_dev(1, double(0.0));
			double *energy_ptr = thrust::raw_pointer_cast(energy_dev.data());

			compute_energy_gpu_aux<double><<<grid, threads>>>(displacement_dev_ptr, jac_it_dev_ptr, global_data_dev_ptr, da_dev_ptr, grad_dev_ptr, n_bases, bv_N, gc_N, n_pts, size(), lambda, mu, energy_ptr);

			cudaDeviceSynchronize();
			thrust::host_vector<double> energy(energy_dev.begin(), energy_dev.end());
			return energy[0];
		}

		Eigen::VectorXd
		NeoHookeanElasticity::assemble_grad_GPU(double *displacement_dev_ptr,
												Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> *jac_it_dev_ptr,
												Local2Global *global_data_dev_ptr,
												Eigen::Matrix<double, -1, 1, 0, 3, 1> *da_dev_ptr,
												Eigen::Matrix<double, -1, -1, 0, 3, 3> *grad_dev_ptr,
												int n_bases,
												int bv_N,
												int gc_N,
												int n_pts,
												double *lambda,
												double *mu,
												int n_basis) const
		{
			int grid = (n_bases % NUMBER_THREADS == 0) ? n_bases / NUMBER_THREADS : n_bases / NUMBER_THREADS + 1;
			int threads = (n_bases > NUMBER_THREADS) ? NUMBER_THREADS : n_bases;

			thrust::device_vector<double> vec_dev(n_basis * size(), double(0.0));
			double *vec_ptr = thrust::raw_pointer_cast(vec_dev.data());
			assert(size() == 3);
			if (bv_N == 4)
			{
				compute_energy_aux_gradient_fast_GPU<4, 3><<<grid, threads, n_basis * size() * sizeof(double)>>>(n_basis, displacement_dev_ptr, jac_it_dev_ptr, global_data_dev_ptr, da_dev_ptr, grad_dev_ptr, n_bases, bv_N, gc_N, n_pts, size(), lambda, mu, vec_ptr);
				cudaDeviceSynchronize();
				thrust::host_vector<double> vec_stg(vec_dev.begin(), vec_dev.end());
				Eigen::Matrix<double, -1, 1> gradient(Eigen::Map<Eigen::Matrix<double, -1, 1>>(vec_stg.data(), vec_stg.size()));
				return gradient;
			}
		}
	} // namespace assembler
} // namespace polyfem