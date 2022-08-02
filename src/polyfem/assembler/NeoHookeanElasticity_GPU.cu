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

		__device__ int kernel_mapping(int *outer, int size_outer, int *inner, int size_inner, int i, int j)
		{
			int index = -1;
			int last_col = 0;

			// ADD SOME SECURITY CHECK LIKE I J >= 0

			for (size_t it_col = 0; it_col < j; ++it_col)
			{
				const auto start = outer[it_col];
				const auto end = outer[it_col + 1];

				for (size_t ii = start; ii < end; ++ii)
				{
					const auto it_row = inner[ii];
					++index;
				}
				++last_col;
			}

			do
			{
				const auto start = outer[last_col];
				const auto end = outer[last_col + 1];

				for (size_t ii = start; ii < end; ++ii)
				{
					const auto it_row = inner[ii];
					if (it_row == i)
					{
						++index;
						break;
					}
					++index;
				}
				//++last_col;
			} while (last_col < j);

			return index;
		}

		template <typename T>
		__device__ void kernel_det(T &mat, double &result)
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

		template <int dim>
		__device__ Eigen::Matrix<double, dim, dim> kernel_hat(const Eigen::Matrix<double, dim, 1> &x)
		{

			Eigen::Matrix<double, dim, dim> prod;
			prod.setZero();

			prod(0, 1) = -x(2);
			prod(0, 2) = x(1);
			prod(1, 0) = x(2);
			prod(1, 2) = -x(0);
			prod(2, 0) = -x(1);
			prod(2, 1) = x(0);

			return prod;
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
					kernel_det<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, dim, dim>>(def_grad, J);
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

		template <int n_basis, int dim, typename T>
		__global__ void compute_energy_hessian_aux_fast_GPU(int N_basis_global,
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
															int *outer,
															int size_outer,
															int *inner,
															int size_inner,
															T *hessian,
															double *computed_values)
		{
			//constexpr int N = (n_basis == Eigen::Dynamic) ? Eigen::Dynamic : n_basis * dim;
			//const int n_pts = da.size();
			Eigen::Matrix<double, -1, -1> thread_values(size_inner, 1);
			thread_values.setZero();

			extern __shared__ double shared_values[];

			int bx = blockIdx.x;
			int tx = threadIdx.x;
			int b_index = bx * NUMBER_THREADS + tx;

			//EACH THREAD SHOULD HAVE ITS OWN HESSIAN
			T H(n_basis * dim, n_basis * dim);
			H.setZero();

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
					kernel_det<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, dim, dim>>(def_grad, J);
					const double log_det_j = log(J);

					Eigen::Matrix<double, dim, dim> delJ_delF(size_, size_);
					delJ_delF.setZero();
					Eigen::Matrix<double, dim * dim, dim * dim> del2J_delF2(size_ * size_, size_ * size_);
					del2J_delF2.setZero();

					if (dim == 2)
					{
						delJ_delF(0, 0) = def_grad(1, 1);
						delJ_delF(0, 1) = -def_grad(1, 0);
						delJ_delF(1, 0) = -def_grad(0, 1);
						delJ_delF(1, 1) = def_grad(0, 0);

						del2J_delF2(0, 3) = 1;
						del2J_delF2(1, 2) = -1;
						del2J_delF2(2, 1) = -1;
						del2J_delF2(3, 0) = 1;
					}
					else if (size_ == 3)
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
						// CHECK BLOCK TEMPLATE BEHAVIOUR IN CUDA KERNEL
						del2J_delF2.template block<dim, dim>(0, 6) = kernel_hat<dim>(v);
						del2J_delF2.template block<dim, dim>(6, 0) = -kernel_hat<dim>(v);
						del2J_delF2.template block<dim, dim>(0, 3) = -kernel_hat<dim>(w);
						del2J_delF2.template block<dim, dim>(3, 0) = kernel_hat<dim>(w);
						del2J_delF2.template block<dim, dim>(3, 6) = -kernel_hat<dim>(u);
						del2J_delF2.template block<dim, dim>(6, 3) = kernel_hat<dim>(u);
					}

					//CHECK THE ID AND MAPPING
					Eigen::Matrix<double, dim * dim, dim *dim> id = Eigen::Matrix<double, dim * dim, dim * dim>::Identity(size_ * size_, size_ * size_);

					Eigen::Matrix<double, dim * dim, 1> g_j = Eigen::Map<const Eigen::Matrix<double, dim * dim, 1>>(delJ_delF.data(), delJ_delF.size());

					Eigen::Matrix<double, dim * dim, dim *dim> hessian_temp = (mu[p] * id) + (((mu[p] + lambda[p] * (1 - log_det_j)) / (J * J)) * (g_j * g_j.transpose())) + (((lambda[p] * log_det_j - mu[p]) / (J)) * del2J_delF2);

					//NOT DYNAMIC YET (n_basis * dim <--> N)
					Eigen::Matrix<double, dim * dim, n_basis * dim> delF_delU_tensor(jac_it.size(), grad.size());

					for (size_t i = 0; i < local_disp.rows(); ++i)
					{
						for (size_t j = 0; j < local_disp.cols(); ++j)
						{
							Eigen::Matrix<double, dim, dim> temp(size_, size_);
							temp.setZero();
							temp.row(j) = grad.row(i);
							temp = temp * jac_it;
							Eigen::Matrix<double, dim * dim, 1> temp_flattened(Eigen::Map<Eigen::Matrix<double, dim * dim, 1>>(temp.data(), temp.size()));
							delF_delU_tensor.col(i * size_ + j) = temp_flattened;
						}
					}

					//NOT DYNAMIC YET (n_basis * dim <--> N)
					Eigen::Matrix<double, n_basis * dim, n_basis *dim> hessian = delF_delU_tensor.transpose() * hessian_temp * delF_delU_tensor;

					double val = mu[p] / 2 * ((def_grad.transpose() * def_grad).trace() - size_ - 2 * log_det_j) + lambda[p] / 2 * log_det_j * log_det_j;

					H += hessian * da[b_index](p);
				}
				//syncthreads
				for (int i = 0; i < bv_N; ++i)
				{
					for (int j = 0; j < bv_N; ++j)
					{
						for (int n = 0; n < size_; ++n)
						{
							for (int m = 0; m < size_; ++m)
							{
								const double local_value = H(i * size_ + m, j * size_ + n);

								for (size_t ii = 0; ii < gc_N; ++ii)
								{
									const auto gi = global_data[b_index * bv_N * gc_N + i * gc_N + ii].index * size_ + m;
									const auto wi = global_data[b_index * bv_N * gc_N + i * gc_N + ii].val;
									for (size_t jj = 0; jj < gc_N; ++jj)
									{
										const auto gj = global_data[b_index * bv_N * gc_N + j * gc_N + jj].index * size_ + n;
										const auto wj = global_data[b_index * bv_N * gc_N + j * gc_N + jj].val;
										const auto val_index = kernel_mapping(outer, size_outer, inner, size_inner, gi, gj);

										thread_values(val_index) += local_value * wi * wj;
									}
								}
							}
						}
					}
				}

				typedef cub::BlockReduce<double, 32> BlockReduce;
				// Allocate shared memory for BlockReduce
				__shared__ typename BlockReduce::TempStorage temp_storage;

				for (int i = 0; i < size_inner; i++)
				{
					double result_ = BlockReduce(temp_storage).Sum(thread_values(i));
					if (tx == 0)
					{
						shared_values[i] = result_;
						atomicAdd(&computed_values[i], shared_values[i]);
					}
				}
			}
		}

		__global__ void print_test(int *outer, int size_outer, int *inner, int size_inner)
		{
			int n1 = 0, n2 = 0, n3 = 0, n4 = 0;
			n1 = kernel_mapping(outer, size_outer, inner, size_inner, 0, 1);
			printf("\n %d %d %d %d\n", n1, n2, n3, n4);
		}

		__global__ void test_values_ext(double *values)
		{
			for (int i = 0; i < 2700; i++)
				values[i] = 69 + i;
		}

		void NeoHookeanElasticity::print_test_wrapper(int *outer, int size_outer, int *inner, int size_inner) const
		{
			print_test<<<1, 1>>>(outer, size_outer, inner, size_inner);
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
			if (size() == 2)
			{
				if (bv_N == 3)
				{
					compute_energy_aux_gradient_fast_GPU<3, 2><<<grid, threads, n_basis * size() * sizeof(double)>>>(n_basis, displacement_dev_ptr, jac_it_dev_ptr, global_data_dev_ptr, da_dev_ptr, grad_dev_ptr, n_bases, bv_N, gc_N, n_pts, size(), lambda, mu, vec_ptr);
				}
				else if (bv_N == 6)
				{
					compute_energy_aux_gradient_fast_GPU<6, 2><<<grid, threads, n_basis * size() * sizeof(double)>>>(n_basis, displacement_dev_ptr, jac_it_dev_ptr, global_data_dev_ptr, da_dev_ptr, grad_dev_ptr, n_bases, bv_N, gc_N, n_pts, size(), lambda, mu, vec_ptr);
				}
				else if (bv_N == 10)
				{
					compute_energy_aux_gradient_fast_GPU<10, 2><<<grid, threads, n_basis * size() * sizeof(double)>>>(n_basis, displacement_dev_ptr, jac_it_dev_ptr, global_data_dev_ptr, da_dev_ptr, grad_dev_ptr, n_bases, bv_N, gc_N, n_pts, size(), lambda, mu, vec_ptr);
				}
				else
				{
					compute_energy_aux_gradient_fast_GPU<Eigen::Dynamic, 2><<<grid, threads, n_basis * size() * sizeof(double)>>>(n_basis, displacement_dev_ptr, jac_it_dev_ptr, global_data_dev_ptr, da_dev_ptr, grad_dev_ptr, n_bases, bv_N, gc_N, n_pts, size(), lambda, mu, vec_ptr);
				}
			}
			else //if (size() == 3)
			{
				assert(size() == 3);
				if (bv_N == 4)
				{
					compute_energy_aux_gradient_fast_GPU<4, 3><<<grid, threads, n_basis * size() * sizeof(double)>>>(n_basis, displacement_dev_ptr, jac_it_dev_ptr, global_data_dev_ptr, da_dev_ptr, grad_dev_ptr, n_bases, bv_N, gc_N, n_pts, size(), lambda, mu, vec_ptr);
				}
				else if (bv_N == 10)
				{
					compute_energy_aux_gradient_fast_GPU<10, 3><<<grid, threads, n_basis * size() * sizeof(double)>>>(n_basis, displacement_dev_ptr, jac_it_dev_ptr, global_data_dev_ptr, da_dev_ptr, grad_dev_ptr, n_bases, bv_N, gc_N, n_pts, size(), lambda, mu, vec_ptr);
				}
				else if (bv_N == 20)
				{
					compute_energy_aux_gradient_fast_GPU<20, 3><<<grid, threads, n_basis * size() * sizeof(double)>>>(n_basis, displacement_dev_ptr, jac_it_dev_ptr, global_data_dev_ptr, da_dev_ptr, grad_dev_ptr, n_bases, bv_N, gc_N, n_pts, size(), lambda, mu, vec_ptr);
				}
				else
				{
					compute_energy_aux_gradient_fast_GPU<Eigen::Dynamic, 3><<<grid, threads, n_basis * size() * sizeof(double)>>>(n_basis, displacement_dev_ptr, jac_it_dev_ptr, global_data_dev_ptr, da_dev_ptr, grad_dev_ptr, n_bases, bv_N, gc_N, n_pts, size(), lambda, mu, vec_ptr);
				}
			}
			cudaDeviceSynchronize();
			thrust::host_vector<double> vec_stg(vec_dev.begin(), vec_dev.end());
			Eigen::Matrix<double, -1, 1> vec(Eigen::Map<Eigen::Matrix<double, -1, 1>>(vec_stg.data(), vec_stg.size()));
			return vec;
		}

		//LETS DO A RETURN
		std::vector<double>
		NeoHookeanElasticity::assemble_hessian_GPU(double *displacement_dev_ptr,
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
												   int n_basis,
												   int *outer_index,
												   int size_outer,
												   int *inner_index,
												   int size_inner) const
		//									   std::vector<double> &computed_values) const
		{
			std::vector<double> computed_values(size_inner, 0);

			thrust::device_vector<double> computed_values_dev(computed_values.begin(), computed_values.end());
			double *computed_values_ptr = thrust::raw_pointer_cast(computed_values_dev.data());

			int grid = (n_bases % NUMBER_THREADS == 0) ? n_bases / NUMBER_THREADS : n_bases / NUMBER_THREADS + 1;
			int threads = (n_bases > NUMBER_THREADS) ? NUMBER_THREADS : n_bases;

			//Eigen::Matrix<double, -1, -1, 0, x, x>> hessian;
			if (size() == 2)
			{
				if (bv_N == 3)
				{
					Eigen::Matrix<double, -1, -1, 0, 6, 6> hessian;
					hessian.setZero();
					thrust::device_vector<Eigen::Matrix<double, -1, -1, 0, 6, 6>> hessian_dev(1);
					hessian_dev[0] = hessian;
					Eigen::Matrix<double, -1, -1, 0, 6, 6> *hessian_ptr = thrust::raw_pointer_cast(hessian_dev.data());
					compute_energy_hessian_aux_fast_GPU<3, 2><<<grid, threads, size_inner * sizeof(double)>>>(n_basis,
																											  displacement_dev_ptr,
																											  jac_it_dev_ptr,
																											  global_data_dev_ptr,
																											  da_dev_ptr,
																											  grad_dev_ptr,
																											  n_bases,
																											  bv_N,
																											  gc_N,
																											  n_pts,
																											  size(),
																											  lambda,
																											  mu,
																											  outer_index,
																											  size_outer,
																											  inner_index,
																											  size_inner,
																											  hessian_ptr,
																											  computed_values_ptr);
				}
				else if (bv_N == 6)
				{
					Eigen::Matrix<double, -1, -1, 0, 12, 12> hessian;
					hessian.setZero();
					thrust::device_vector<Eigen::Matrix<double, -1, -1, 0, 12, 12>> hessian_dev(1);
					hessian_dev[0] = hessian;
					Eigen::Matrix<double, -1, -1, 0, 12, 12> *hessian_ptr = thrust::raw_pointer_cast(hessian_dev.data());
					compute_energy_hessian_aux_fast_GPU<6, 2><<<grid, threads, size_inner * sizeof(double)>>>(n_basis,
																											  displacement_dev_ptr,
																											  jac_it_dev_ptr,
																											  global_data_dev_ptr,
																											  da_dev_ptr,
																											  grad_dev_ptr,
																											  n_bases,
																											  bv_N,
																											  gc_N,
																											  n_pts,
																											  size(),
																											  lambda,
																											  mu,
																											  outer_index,
																											  size_outer,
																											  inner_index,
																											  size_inner,
																											  hessian_ptr,
																											  computed_values_ptr);
				}
				else if (bv_N == 10)
				{
					Eigen::Matrix<double, -1, -1, 0, 20, 20> hessian;
					hessian.setZero();
					thrust::device_vector<Eigen::Matrix<double, -1, -1, 0, 20, 20>> hessian_dev(1);
					hessian_dev[0] = hessian;
					Eigen::Matrix<double, -1, -1, 0, 20, 20> *hessian_ptr = thrust::raw_pointer_cast(hessian_dev.data());
					compute_energy_hessian_aux_fast_GPU<10, 2><<<grid, threads, size_inner * sizeof(double)>>>(n_basis,
																											   displacement_dev_ptr,
																											   jac_it_dev_ptr,
																											   global_data_dev_ptr,
																											   da_dev_ptr,
																											   grad_dev_ptr,
																											   n_bases,
																											   bv_N,
																											   gc_N,
																											   n_pts,
																											   size(),
																											   lambda,
																											   mu,
																											   outer_index,
																											   size_outer,
																											   inner_index,
																											   size_inner,
																											   hessian_ptr,
																											   computed_values_ptr);
				}
			}
			else
			{
				assert(size() == 3);
				if (bv_N == 4)
				{
					Eigen::Matrix<double, -1, -1, 0, 12, 12> hessian;
					hessian.setZero();
					thrust::device_vector<Eigen::Matrix<double, -1, -1, 0, 12, 12>> hessian_dev(1);
					hessian_dev[0] = hessian;
					Eigen::Matrix<double, -1, -1, 0, 12, 12> *hessian_ptr = thrust::raw_pointer_cast(hessian_dev.data());

					compute_energy_hessian_aux_fast_GPU<4, 3><<<grid, threads, size_inner * sizeof(double)>>>(n_basis,
																											  displacement_dev_ptr,
																											  jac_it_dev_ptr,
																											  global_data_dev_ptr,
																											  da_dev_ptr,
																											  grad_dev_ptr,
																											  n_bases,
																											  bv_N,
																											  gc_N,
																											  n_pts,
																											  size(),
																											  lambda,
																											  mu,
																											  outer_index,
																											  size_outer,
																											  inner_index,
																											  size_inner,
																											  hessian_ptr,
																											  computed_values_ptr);
				}
				else if (bv_N == 10)
				{
					Eigen::Matrix<double, -1, -1, 0, 30, 30> hessian;
					hessian.setZero();
					thrust::device_vector<Eigen::Matrix<double, -1, -1, 0, 30, 30>> hessian_dev(1);
					hessian_dev[0] = hessian;
					Eigen::Matrix<double, -1, -1, 0, 30, 30> *hessian_ptr = thrust::raw_pointer_cast(hessian_dev.data());

					compute_energy_hessian_aux_fast_GPU<10, 3><<<grid, threads, size_inner * sizeof(double)>>>(n_basis,
																											   displacement_dev_ptr,
																											   jac_it_dev_ptr,
																											   global_data_dev_ptr,
																											   da_dev_ptr,
																											   grad_dev_ptr,
																											   n_bases,
																											   bv_N,
																											   gc_N,
																											   n_pts,
																											   size(),
																											   lambda,
																											   mu,
																											   outer_index,
																											   size_outer,
																											   inner_index,
																											   size_inner,
																											   hessian_ptr,
																											   computed_values_ptr);
				}
				else if (bv_N == 20)
				{
					Eigen::Matrix<double, -1, -1, 0, 60, 60> hessian;
					hessian.setZero();
					thrust::device_vector<Eigen::Matrix<double, -1, -1, 0, 60, 60>> hessian_dev(1);
					hessian_dev[0] = hessian;
					Eigen::Matrix<double, -1, -1, 0, 60, 60> *hessian_ptr = thrust::raw_pointer_cast(hessian_dev.data());

					compute_energy_hessian_aux_fast_GPU<20, 3><<<grid, threads, size_inner * sizeof(double)>>>(n_basis,
																											   displacement_dev_ptr,
																											   jac_it_dev_ptr,
																											   global_data_dev_ptr,
																											   da_dev_ptr,
																											   grad_dev_ptr,
																											   n_bases,
																											   bv_N,
																											   gc_N,
																											   n_pts,
																											   size(),
																											   lambda,
																											   mu,
																											   outer_index,
																											   size_outer,
																											   inner_index,
																											   size_inner,
																											   hessian_ptr,
																											   computed_values_ptr);
				}
			}
			cudaDeviceSynchronize();
			thrust::copy(computed_values_dev.begin(), computed_values_dev.end(), computed_values.begin());
			// empty the vector
			//computed_values_dev.clear();

			// deallocate any capacity which may currently be associated with vec
			//computed_values.shrink_to_fit();
			return computed_values;
		}

	} // namespace assembler

	//} // namespace assembler
} // namespace polyfem