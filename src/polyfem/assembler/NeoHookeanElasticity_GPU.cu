#include "NeoHookeanElasticity.hpp"
#include <polyfem/basis/Basis.hpp>
#include <polyfem/autogen/auto_elasticity_rhs.hpp>

#include <polyfem/utils/CUDA_utilities.cuh>

#include <polyfem/utils/MatrixUtils.hpp>
#include <igl/Timer.h>

namespace polyfem
{
	using namespace basis;

	namespace assembler
	{

		// ADD A MAX _ VAL OR STOP CONDITION FOR SECURITY
		__device__ int kernel_mapping(mapping_pair **mapping, int i, int j)
		{
			int k = 0;
			do
			{
				if (mapping[i][k].first == j)
				{
					return mapping[i][k].second;
				}
				k++;
			} while (1);
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

		template <int n_basis, int dim>
		__global__ void compute_energy_gpu_aux(double *displacement,
											   Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> *jac_it_array,
											   Local2Global_GPU *global_data,
											   Eigen::Matrix<double, -1, 1, 0, 4, 1> *da,
											   Eigen::Matrix<double, -1, -1, 0, 4, 3> *grad_v,
											   int n_bases,
											   int n_loc_bases,
											   int global_vector_size,
											   int n_pts,
											   int _size,
											   double *lambda,
											   double *mu,
											   double *energy_val)
		{
			int bx = blockIdx.x;
			int tx = threadIdx.x;
			int b_index = bx * NUMBER_THREADS + tx;
			constexpr int N = (n_basis == Eigen::Dynamic) ? Eigen::Dynamic : n_basis * dim;
			if (b_index < n_bases)
			{
				Eigen::Matrix<double, N, 1> local_dispv(n_loc_bases * _size, 1);

				local_dispv.setZero();
				for (int i = 0; i < n_loc_bases; ++i)
				{
					for (int ii = 0; ii < global_vector_size; ++ii)
					{
						for (int d = 0; d < _size; ++d)
						{
							// threads allocation jumps the size of the respective vectors
							local_dispv(i * _size + d) += global_data[b_index * n_loc_bases * global_vector_size + i * global_vector_size + ii].val * displacement[global_data[b_index * n_loc_bases * global_vector_size + i * global_vector_size + ii].index * _size + d];
						}
					}
				}

				double energy = double(0.0);
				Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> def_grad(_size, _size);

				for (long p = 0; p < n_pts; ++p)
				{
					for (long k = 0; k < def_grad.size(); ++k)
						def_grad(k) = double(0);

					for (size_t i = 0; i < n_loc_bases; ++i)
					{
						for (int d = 0; d < _size; ++d)
						{
							for (int c = 0; c < _size; ++c)
							{
								double val_grad = grad_v[b_index * n_loc_bases + i].row(p)(c);
								def_grad(d, c) += val_grad * local_dispv(i * _size + d);
							}
						}
					}
					Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> jac_it(_size, _size);
					for (long k = 0; k < jac_it.size(); ++k)
						jac_it(k) = double(jac_it_array[b_index * n_pts + p](k));
					def_grad = def_grad * jac_it;

					// Id + grad d
					for (int d = 0; d < _size; ++d)
						def_grad(d, d) += double(1);

					double _det;
					kernel_det<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>>(def_grad, _det);
					const double log_det_j = log(_det);
					const double val = mu[b_index * n_pts + p] / 2 * ((def_grad.transpose() * def_grad).trace() - _size - 2 * log_det_j) + lambda[b_index * n_pts + p] / 2 * log_det_j * log_det_j;
					energy += val * da[b_index](p);
				}

				atomicAdd(&energy_val[0], energy);
				__syncthreads();
			}
		}

		template <int n_basis, int dim>
		__global__ void compute_energy_aux_gradient_fast_GPU(int n_basis_global,
															 double *displacement,
															 Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> *jac_it_array,
															 Local2Global_GPU *global_data,
															 Eigen::Matrix<double, -1, 1, 0, 4, 1> *da,
															 Eigen::Matrix<double, -1, -1, 0, 4, 3> *grad_v,
															 int n_bases,
															 int n_loc_bases,
															 int global_vector_size,
															 int n_pts,
															 int size_,
															 double *lambda,
															 double *mu,
															 double *result_vec)
		{
			int bx = blockIdx.x;
			int tx = threadIdx.x;
			int b_index = bx * NUMBER_THREADS + tx;

			if (b_index < n_bases)
			{
				Eigen::Matrix<double, n_basis, dim> local_disp(n_loc_bases, size_);
				local_disp.setZero();
				for (int i = 0; i < n_loc_bases; ++i)
				{
					for (int ii = 0; ii < global_vector_size; ++ii)
					{
						for (int d = 0; d < size_; ++d)
						{
							// threads allocation jumps the size of the respective vectors
							local_disp(i, d) += global_data[b_index * n_loc_bases * global_vector_size + i * global_vector_size + ii].val * displacement[global_data[b_index * n_loc_bases * global_vector_size + i * global_vector_size + ii].index * size_ + d];
						}
					}
				}
				Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, dim, dim> def_grad(size_, size_);

				Eigen::Matrix<double, n_basis, dim> G(n_loc_bases, size_);
				G.setZero();

				for (long p = 0; p < n_pts; ++p)
				{
					Eigen::Matrix<double, n_basis, dim> grad(n_loc_bases, size_);
					for (size_t i = 0; i < n_loc_bases; ++i)
					{
						if (n_loc_bases == 4 || n_loc_bases == 3)
							grad_v[b_index * n_loc_bases + i].resize(1, 3);
						grad.row(i) = grad_v[b_index * n_loc_bases + i].row(p);
					}
					Eigen::Matrix<double, dim, dim> jac_it;
					for (long k = 0; k < jac_it.size(); ++k)
						jac_it(k) = jac_it_array[b_index * n_pts + p](k);

					// Id + grad d
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

					Eigen::Matrix<double, dim, dim> gradient_temp = mu[b_index * n_pts + p] * def_grad - mu[b_index * n_pts + p] * (1 / J) * delJ_delF + lambda[b_index * n_pts + p] * log_det_j * (1 / J) * delJ_delF;
					Eigen::Matrix<double, n_basis, dim> gradient = delF_delU * gradient_temp.transpose();

					double val = mu[b_index * n_pts + p] / 2 * ((def_grad.transpose() * def_grad).trace() - size_ - 2 * log_det_j) + lambda[b_index * n_pts + p] / 2 * log_det_j * log_det_j;

					G.noalias() += gradient * da[b_index](p);
				}

				Eigen::Matrix<double, dim, n_basis> G_T = G.transpose();

				constexpr int N = (n_basis == Eigen::Dynamic) ? Eigen::Dynamic : n_basis * dim;
				Eigen::Matrix<double, N, 1> temp(Eigen::Map<Eigen::Matrix<double, N, 1>>(G_T.data(), G_T.size()));

				for (int j = 0; j < n_loc_bases; ++j)
				{
					for (int m = 0; m < size_; ++m)
					{
						const double local_value = temp(j * size_ + m);
						if (std::abs(local_value) < 1e-30)
						{
							continue;
						}

						for (size_t jj = 0; jj < global_vector_size; ++jj)
						{
							const auto gj = global_data[b_index * n_loc_bases * global_vector_size + j * global_vector_size + jj].index * size_ + m;
							const auto wj = global_data[b_index * n_loc_bases * global_vector_size + j * global_vector_size + jj].val;
							atomicAdd(&result_vec[gj], local_value * wj);
						}
					}
				}
				__syncthreads();
			}
		}

		// Missing implementation for dynamic number of basis
		template <int n_basis, int dim>
		__global__ void compute_energy_hessian_aux_fast_GPU(int it_index,
															double *displacement,
															Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> *jac_it_array,
															Local2Global_GPU *global_data,
															Eigen::Matrix<double, -1, 1, 0, 4, 1> *da,
															Eigen::Matrix<double, -1, -1, 0, 4, 3> *grad_v,
															int n_bases,
															int partial_bases,
															int n_loc_bases,
															int global_vector_size,
															int n_pts,
															int size_,
															double *lambda,
															double *mu,
															//	mapping_pair **mapping,
															int **second_cache,
															double *computed_values)
		{
			// constexpr int N = (n_basis == Eigen::Dynamic) ? Eigen::Dynamic : n_basis * dim;

			int bx = blockIdx.x;
			int tx = threadIdx.x;
			int t_index = bx * NUMBER_THREADS + tx;
			int b_index = bx * NUMBER_THREADS + tx + it_index * partial_bases;

			int thread_boundary = ((n_bases - partial_bases * (it_index + 2)) >= 0) ? partial_bases : n_bases - partial_bases * it_index;

			if (b_index < n_bases && t_index < thread_boundary)
			{
				Eigen::Matrix<double, -1, -1, 0, n_basis * dim, n_basis * dim> H(n_basis * dim, n_basis * dim);
				H.setZero();

				Eigen::Matrix<double, n_basis, dim> local_disp(n_loc_bases, size_);
				local_disp.setZero();

				for (int i = 0; i < n_loc_bases; ++i)
				{
					for (int ii = 0; ii < global_vector_size; ++ii)
					{
						for (int d = 0; d < size_; ++d)
						{
							// threads allocation jumps the size of the respective vectors
							local_disp(i, d) += global_data[b_index * n_loc_bases * global_vector_size + i * global_vector_size + ii].val * displacement[global_data[b_index * n_loc_bases * global_vector_size + i * global_vector_size + ii].index * size_ + d];
						}
					}
				}
				Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, dim, dim> def_grad(size_, size_);

				for (long p = 0; p < n_pts; ++p)
				{
					Eigen::Matrix<double, n_basis, dim> grad(n_loc_bases, size_);
					for (size_t i = 0; i < n_loc_bases; ++i)
					{
						if (n_loc_bases == 4 || n_loc_bases == 3)
							grad_v[b_index * n_loc_bases + i].resize(1, 3);
						// grad.row(i) = grad_v[b_index * n_loc_bases + i].col(p); // WORKS ONLY FOR LINEAR, MAYBE WE SHOULD FIND ANOTHER WAY
						//	else
						grad.row(i) = grad_v[b_index * n_loc_bases + i].row(p);
					}
					Eigen::Matrix<double, dim, dim> jac_it;
					for (long k = 0; k < jac_it.size(); ++k)
						jac_it(k) = jac_it_array[b_index * n_pts + p](k);

					// Id + grad d
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

						// PASSED TESTS FOR N_BASIS = 4 , 10
						del2J_delF2.template block<dim, dim>(0, 6) = kernel_hat<dim>(v);
						del2J_delF2.template block<dim, dim>(6, 0) = -kernel_hat<dim>(v);
						del2J_delF2.template block<dim, dim>(0, 3) = -kernel_hat<dim>(w);
						del2J_delF2.template block<dim, dim>(3, 0) = kernel_hat<dim>(w);
						del2J_delF2.template block<dim, dim>(3, 6) = -kernel_hat<dim>(u);
						del2J_delF2.template block<dim, dim>(6, 3) = kernel_hat<dim>(u);
					}

					// CHECK THE ID AND MAPPING
					Eigen::Matrix<double, dim * dim, dim *dim> id = Eigen::Matrix<double, dim * dim, dim * dim>::Identity(size_ * size_, size_ * size_);

					Eigen::Matrix<double, dim * dim, 1> g_j = Eigen::Map<const Eigen::Matrix<double, dim * dim, 1>>(delJ_delF.data(), delJ_delF.size());

					Eigen::Matrix<double, dim * dim, dim *dim> hessian_temp = (mu[b_index * n_pts + p] * id) + (((mu[b_index * n_pts + p] + lambda[b_index * n_pts + p] * (1 - log_det_j)) / (J * J)) * (g_j * g_j.transpose())) + (((lambda[b_index * n_pts + p] * log_det_j - mu[b_index * n_pts + p]) / (J)) * del2J_delF2);

					// NOT DYNAMIC YET (n_basis * dim <--> N)
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

					// NOT DYNAMIC YET (n_basis * dim <--> N)
					Eigen::Matrix<double, n_basis * dim, n_basis *dim> hessian = delF_delU_tensor.transpose() * hessian_temp * delF_delU_tensor;

					double val = mu[b_index * n_pts + p] / 2 * ((def_grad.transpose() * def_grad).trace() - size_ - 2 * log_det_j) + lambda[b_index * n_pts + p] / 2 * log_det_j * log_det_j;

					H += hessian * da[b_index](p);
				}

				// syncthreads?
				int it = 0;
				for (int i = 0; i < n_loc_bases; ++i)
				{
					for (int j = 0; j < n_loc_bases; ++j)
					{
						for (int n = 0; n < size_; ++n)
						{
							for (int m = 0; m < size_; ++m)
							{
								const double local_value = H(i * size_ + m, j * size_ + n);

								for (size_t ii = 0; ii < global_vector_size; ++ii)
								{
									const auto gi = global_data[b_index * n_loc_bases * global_vector_size + i * global_vector_size + ii].index * size_ + m;
									const auto wi = global_data[b_index * n_loc_bases * global_vector_size + i * global_vector_size + ii].val;
									for (size_t jj = 0; jj < global_vector_size; ++jj)
									{
										const auto gj = global_data[b_index * n_loc_bases * global_vector_size + j * global_vector_size + jj].index * size_ + n;
										const auto wj = global_data[b_index * n_loc_bases * global_vector_size + j * global_vector_size + jj].val;
										// const auto val_index = kernel_mapping(mapping, gi, gj);
										const auto val_index = second_cache[b_index][it];
										it++;
										atomicAdd(&computed_values[val_index], local_value * wi * wj);
									}
								}
							}
						}
					}
				}
			}
		}

		double NeoHookeanElasticity::compute_energy_gpu(double *displacement_dev_ptr,
														Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> *jac_it_dev_ptr,
														Local2Global_GPU *global_data_dev_ptr,
														Eigen::Matrix<double, -1, 1, 0, 4, 1> *da_dev_ptr,
														Eigen::Matrix<double, -1, -1, 0, 4, 3> *grad_dev_ptr,
														int n_bases,
														int n_loc_bases,
														int global_vector_size,
														int n_pts,
														double *lambda,
														double *mu) const
		{
			int grid = (n_bases % NUMBER_THREADS == 0) ? n_bases / NUMBER_THREADS : n_bases / NUMBER_THREADS + 1;
			int threads = (n_bases > NUMBER_THREADS) ? NUMBER_THREADS : n_bases;

			thrust::device_vector<double> energy_dev(1, double(0.0));
			double *energy_ptr = thrust::raw_pointer_cast(energy_dev.data());
			if (size() == 2)
			{
				if (n_loc_bases == 3)
				{
					compute_energy_gpu_aux<3, 2><<<grid, threads>>>(displacement_dev_ptr, jac_it_dev_ptr, global_data_dev_ptr, da_dev_ptr, grad_dev_ptr, n_bases, n_loc_bases, global_vector_size, n_pts, size(), lambda, mu, energy_ptr);
				}
				else if (n_loc_bases == 6)
				{
					compute_energy_gpu_aux<6, 2><<<grid, threads>>>(displacement_dev_ptr, jac_it_dev_ptr, global_data_dev_ptr, da_dev_ptr, grad_dev_ptr, n_bases, n_loc_bases, global_vector_size, n_pts, size(), lambda, mu, energy_ptr);
				}
				else if (n_loc_bases == 10)
				{
					compute_energy_gpu_aux<10, 2><<<grid, threads>>>(displacement_dev_ptr, jac_it_dev_ptr, global_data_dev_ptr, da_dev_ptr, grad_dev_ptr, n_bases, n_loc_bases, global_vector_size, n_pts, size(), lambda, mu, energy_ptr);
				}
				else
				{
					compute_energy_gpu_aux<Eigen::Dynamic, 2><<<grid, threads>>>(displacement_dev_ptr, jac_it_dev_ptr, global_data_dev_ptr, da_dev_ptr, grad_dev_ptr, n_bases, n_loc_bases, global_vector_size, n_pts, size(), lambda, mu, energy_ptr);
				}
			}
			else // if (size() == 3)
			{
				assert(size() == 3);
				if (n_loc_bases == 4)
				{
					compute_energy_gpu_aux<4, 3><<<grid, threads>>>(displacement_dev_ptr, jac_it_dev_ptr, global_data_dev_ptr, da_dev_ptr, grad_dev_ptr, n_bases, n_loc_bases, global_vector_size, n_pts, size(), lambda, mu, energy_ptr);
				}
				else if (n_loc_bases == 10)
				{
					compute_energy_gpu_aux<10, 3><<<grid, threads>>>(displacement_dev_ptr, jac_it_dev_ptr, global_data_dev_ptr, da_dev_ptr, grad_dev_ptr, n_bases, n_loc_bases, global_vector_size, n_pts, size(), lambda, mu, energy_ptr);
				}
				else if (n_loc_bases == 20)
				{
					compute_energy_gpu_aux<20, 3><<<grid, threads>>>(displacement_dev_ptr, jac_it_dev_ptr, global_data_dev_ptr, da_dev_ptr, grad_dev_ptr, n_bases, n_loc_bases, global_vector_size, n_pts, size(), lambda, mu, energy_ptr);
				}
				else
				{
					compute_energy_gpu_aux<Eigen::Dynamic, 3><<<grid, threads>>>(displacement_dev_ptr, jac_it_dev_ptr, global_data_dev_ptr, da_dev_ptr, grad_dev_ptr, n_bases, n_loc_bases, global_vector_size, n_pts, size(), lambda, mu, energy_ptr);
				}
			}

			gpuErrchk(cudaPeekAtLastError());
			CHECK_CUDA_ERROR(cudaDeviceSynchronize());
			thrust::host_vector<double> energy(energy_dev.begin(), energy_dev.end());
			return energy[0];
		}

		Eigen::VectorXd
		NeoHookeanElasticity::assemble_grad_GPU(double *displacement_dev_ptr,
												Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> *jac_it_dev_ptr,
												Local2Global_GPU *global_data_dev_ptr,
												Eigen::Matrix<double, -1, 1, 0, 4, 1> *da_dev_ptr,
												Eigen::Matrix<double, -1, -1, 0, 4, 3> *grad_dev_ptr,
												int n_bases,
												int n_loc_bases,
												int global_vector_size,
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
				if (n_loc_bases == 3)
				{
					compute_energy_aux_gradient_fast_GPU<3, 2><<<grid, threads>>>(n_basis, displacement_dev_ptr, jac_it_dev_ptr, global_data_dev_ptr, da_dev_ptr, grad_dev_ptr, n_bases, n_loc_bases, global_vector_size, n_pts, size(), lambda, mu, vec_ptr);
				}
				else if (n_loc_bases == 6)
				{
					compute_energy_aux_gradient_fast_GPU<6, 2><<<grid, threads>>>(n_basis, displacement_dev_ptr, jac_it_dev_ptr, global_data_dev_ptr, da_dev_ptr, grad_dev_ptr, n_bases, n_loc_bases, global_vector_size, n_pts, size(), lambda, mu, vec_ptr);
				}
				else if (n_loc_bases == 10)
				{
					compute_energy_aux_gradient_fast_GPU<10, 2><<<grid, threads>>>(n_basis, displacement_dev_ptr, jac_it_dev_ptr, global_data_dev_ptr, da_dev_ptr, grad_dev_ptr, n_bases, n_loc_bases, global_vector_size, n_pts, size(), lambda, mu, vec_ptr);
				}
				else
				{
					compute_energy_aux_gradient_fast_GPU<Eigen::Dynamic, 2><<<grid, threads>>>(n_basis, displacement_dev_ptr, jac_it_dev_ptr, global_data_dev_ptr, da_dev_ptr, grad_dev_ptr, n_bases, n_loc_bases, global_vector_size, n_pts, size(), lambda, mu, vec_ptr);
				}
			}
			else // if (size() == 3)
			{
				assert(size() == 3);
				if (n_loc_bases == 4)
				{
					compute_energy_aux_gradient_fast_GPU<4, 3><<<grid, threads>>>(n_basis, displacement_dev_ptr, jac_it_dev_ptr, global_data_dev_ptr, da_dev_ptr, grad_dev_ptr, n_bases, n_loc_bases, global_vector_size, n_pts, size(), lambda, mu, vec_ptr);
				}
				else if (n_loc_bases == 10)
				{
					compute_energy_aux_gradient_fast_GPU<10, 3><<<grid, threads>>>(n_basis, displacement_dev_ptr, jac_it_dev_ptr, global_data_dev_ptr, da_dev_ptr, grad_dev_ptr, n_bases, n_loc_bases, global_vector_size, n_pts, size(), lambda, mu, vec_ptr);
				}
				else if (n_loc_bases == 20)
				{
					compute_energy_aux_gradient_fast_GPU<20, 3><<<grid, threads>>>(n_basis, displacement_dev_ptr, jac_it_dev_ptr, global_data_dev_ptr, da_dev_ptr, grad_dev_ptr, n_bases, n_loc_bases, global_vector_size, n_pts, size(), lambda, mu, vec_ptr);
				}
				else
				{
					compute_energy_aux_gradient_fast_GPU<Eigen::Dynamic, 3><<<grid, threads>>>(n_basis, displacement_dev_ptr, jac_it_dev_ptr, global_data_dev_ptr, da_dev_ptr, grad_dev_ptr, n_bases, n_loc_bases, global_vector_size, n_pts, size(), lambda, mu, vec_ptr);
				}
			}
			gpuErrchk(cudaPeekAtLastError());
			CHECK_CUDA_ERROR(cudaDeviceSynchronize());
			thrust::host_vector<double> vec_stg(vec_dev.begin(), vec_dev.end());
			Eigen::Matrix<double, -1, 1> vec(Eigen::Map<Eigen::Matrix<double, -1, 1>>(vec_stg.data(), vec_stg.size()));
			return vec;
		}

		// LETS DO A RETURN
		std::vector<double>
		NeoHookeanElasticity::assemble_hessian_GPU(double *displacement_dev_ptr,
												   Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> *jac_it_dev_ptr,
												   Local2Global_GPU *global_data_dev_ptr,
												   Eigen::Matrix<double, -1, 1, 0, 4, 1> *da_dev_ptr,
												   Eigen::Matrix<double, -1, -1, 0, 4, 3> *grad_dev_ptr,
												   int n_bases,
												   int n_loc_bases,
												   int global_vector_size,
												   int n_pts,
												   double *lambda,
												   double *mu,
												   int non_zeros,
												   //  mapping_pair **mapping,
												   int **second_cache) const
		//									   std::vector<double> &computed_values) const
		{
			std::vector<double> computed_values(non_zeros, 0);

			thrust::device_vector<double> computed_values_dev(computed_values.begin(), computed_values.end());
			double *computed_values_ptr = thrust::raw_pointer_cast(computed_values_dev.data());
			//			long double slice = ((long double)n_bases * size_inner * sizeof(double)) / (12e9);
			long double slice = 0.99; // NEED TO CALCULATE MAX ALLOCATION POSSIBLE
			int iterations = (slice > 1) ? slice + 1 : 1;
			//	int const sharedMemoryBytes{size_inner * sizeof(double)};
			int partial_bases = n_bases / iterations;

			// Limitation of the GPU memory

			// Eigen::Matrix<double, -1, -1, 0, x, x>> hessian;
			for (int it = 0; it < iterations; it++)
			{
				auto threads__ = ((n_bases - partial_bases * (it + 2)) >= 0) ? partial_bases : n_bases - partial_bases * it;
				int grid = (threads__ % NUMBER_THREADS == 0) ? threads__ / NUMBER_THREADS : threads__ / NUMBER_THREADS + 1;
				int threads = (threads__ > NUMBER_THREADS) ? NUMBER_THREADS : threads__;

				if (size() == 2)
				{
					if (n_loc_bases == 3)
					{
						//	Eigen::Matrix<double, -1, -1, 0, 6, 6> hessian;
						//	hessian.setZero();
						//	thrust::device_vector<Eigen::Matrix<double, -1, -1, 0, 6, 6>> hessian_dev(1);
						//	hessian_dev[0] = hessian;
						//	Eigen::Matrix<double, -1, -1, 0, 6, 6> *hessian_ptr = thrust::raw_pointer_cast(hessian_dev.data());
						compute_energy_hessian_aux_fast_GPU<3, 2><<<grid, threads>>>(it,
																					 displacement_dev_ptr,
																					 jac_it_dev_ptr,
																					 global_data_dev_ptr,
																					 da_dev_ptr,
																					 grad_dev_ptr,
																					 n_bases,
																					 partial_bases,
																					 n_loc_bases,
																					 global_vector_size,
																					 n_pts,
																					 size(),
																					 lambda,
																					 mu,
																					 //	 mapping,
																					 second_cache,
																					 computed_values_ptr);
					}
					else if (n_loc_bases == 6)
					{
						compute_energy_hessian_aux_fast_GPU<6, 2><<<grid, threads>>>(it,
																					 displacement_dev_ptr,
																					 jac_it_dev_ptr,
																					 global_data_dev_ptr,
																					 da_dev_ptr,
																					 grad_dev_ptr,
																					 n_bases,
																					 partial_bases,
																					 n_loc_bases,
																					 global_vector_size,
																					 n_pts,
																					 size(),
																					 lambda,
																					 mu,
																					 //	 mapping,
																					 second_cache,
																					 computed_values_ptr);
					}
					else if (n_loc_bases == 10)
					{
						compute_energy_hessian_aux_fast_GPU<10, 2><<<grid, threads>>>(it,
																					  displacement_dev_ptr,
																					  jac_it_dev_ptr,
																					  global_data_dev_ptr,
																					  da_dev_ptr,
																					  grad_dev_ptr,
																					  n_bases,
																					  partial_bases,
																					  n_loc_bases,
																					  global_vector_size,
																					  n_pts,
																					  size(),
																					  lambda,
																					  mu,
																					  //	  mapping,
																					  second_cache,
																					  computed_values_ptr);
					}
				}
				else
				{
					assert(size() == 3);
					if (n_loc_bases == 4)
					{

						// ADD A KERNEL WRAPPER
						// compute_energy_hessian_aux_fast_GPU<4, 3><<<grid, threads, sharedMemoryBytes>>>(displacement_dev_ptr,
						compute_energy_hessian_aux_fast_GPU<4, 3><<<grid, threads>>>(it,
																					 displacement_dev_ptr,
																					 jac_it_dev_ptr,
																					 global_data_dev_ptr,
																					 da_dev_ptr,
																					 grad_dev_ptr,
																					 n_bases,
																					 partial_bases,
																					 n_loc_bases,
																					 global_vector_size,
																					 n_pts,
																					 size(),
																					 lambda,
																					 mu,
																					 //	 mapping,
																					 second_cache,
																					 computed_values_ptr);
					}
					else if (n_loc_bases == 10)
					{
						compute_energy_hessian_aux_fast_GPU<10, 3><<<grid, threads>>>(it,
																					  displacement_dev_ptr,
																					  jac_it_dev_ptr,
																					  global_data_dev_ptr,
																					  da_dev_ptr,
																					  grad_dev_ptr,
																					  n_bases,
																					  partial_bases,
																					  n_loc_bases,
																					  global_vector_size,
																					  n_pts,
																					  size(),
																					  lambda,
																					  mu,
																					  //	  mapping,
																					  second_cache,
																					  computed_values_ptr);
					}
					else if (n_loc_bases == 20)
					{
						compute_energy_hessian_aux_fast_GPU<20, 3><<<grid, threads>>>(it,
																					  displacement_dev_ptr,
																					  jac_it_dev_ptr,
																					  global_data_dev_ptr,
																					  da_dev_ptr,
																					  grad_dev_ptr,
																					  n_bases,
																					  partial_bases,
																					  n_loc_bases,
																					  global_vector_size,
																					  n_pts,
																					  size(),
																					  lambda,
																					  mu,
																					  //	  mapping,
																					  second_cache,
																					  computed_values_ptr);
					}
				}
				gpuErrchk(cudaPeekAtLastError());
				CHECK_CUDA_ERROR(cudaDeviceSynchronize());
			}
			thrust::copy(computed_values_dev.begin(), computed_values_dev.end(), computed_values.begin());

			return computed_values;
		}

	} // namespace assembler

	//} // namespace assembler
} // namespace polyfem