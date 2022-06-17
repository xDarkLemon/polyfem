#include "NeoHookeanElasticity.hpp"
#include <assembler/utils/CUDA_utilities.cuh>
#include <polyfem/basis/Basis.hpp>
#include <polyfem/autogen/auto_elasticity_rhs.hpp>

#include <polyfem/utils/MatrixUtils.hpp>

namespace polyfem
{
	using namespace basis;
	namespace assembler
	{
		__global__ void set_dispv(Local2Global *bvs_data, int *bvs_sizes, double *displacement_dev, int size, int bvs_total_size, double *local_disp)
		{
			int bx = blockIdx.x;
			int tx = threadIdx.x;
			int inner_index = bx * NUMBER_THREADS + tx;
			double result = 0.0;

			if (inner_index < bvs_total_size)
			{
				for (size_t ii = 0; ii < bvs_sizes[inner_index]; ++ii)
				{
					for (size_t ii = 0; ii < bvs_sizes[inner_index]; ++ii)
					{
						for (int d = 0; d < size; ++d)
						{
							result = bvs_data[inner_index + ii * bvs_total_size].val * displacement_dev[bvs_data[inner_index + ii * bvs_total_size].index * size + d];
							local_disp[inner_index * size + d] += result;
						}
					}
				}
			}
		}

		template <typename T>
		__global__ void defgrad_comp(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> *def_grad, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> *grad, double *local_dispv, int size, int i)
		{
			int bx = blockIdx.x;
			int by = blockIdx.y;
			int tx = threadIdx.x;
			int ty = threadIdx.y;
			int inner_index_x = bx * NUMBER_THREADS + tx;
			int inner_index_y = by * NUMBER_THREADS + ty;
			if (inner_index_x < size && inner_index_y < size)
			{
				def_grad[0](inner_index_x, inner_index_y) += grad[0](inner_index_y) * local_dispv[i * size + inner_index_x];
			}
		}

		double NeoHookeanElasticity::compute_energy_GPU(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const
		{
			return compute_energy_aux<double>(vals, displacement, da);
		}

		// Compute ∫ ½μ (tr(FᵀF) - 3 - 2ln(J)) + ½λ ln²(J) du
		template <typename T>
		T NeoHookeanElasticity::compute_energy_aux(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const
		{
			typedef Eigen::Matrix<T, Eigen::Dynamic, 1> AutoDiffVect;
			typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> AutoDiffGradMat;

			assert(displacement.cols() == 1);

			double *local_dispv_dev = NULL;
			double *displacement_dev = NULL;
			int *bs_global_sizes_dev = NULL;

			size_t basisvalues_size = vals.basis_values.size();

			int localdispv_data_size = basisvalues_size * size() * sizeof(double);
			int displ_data_size = displacement.rows() * displacement.cols() * sizeof(double);
			int bsglobalsizes_data_size = basisvalues_size * sizeof(int);

			double *local_dispv = new double[basisvalues_size * size()];

			local_dispv_dev = ALLOCATE_GPU<double>(local_dispv_dev, localdispv_data_size);

			displacement_dev = ALLOCATE_GPU<double>(displacement_dev, displ_data_size);

			bs_global_sizes_dev = ALLOCATE_GPU<int>(bs_global_sizes_dev, bsglobalsizes_data_size);

			COPYDATATOGPU<double>(displacement_dev, displacement.data(), displ_data_size);

			cudaDeviceSynchronize();

			const int n_pts = da.size();

			const AssemblyValues *bs_storage = vals.basis_values.data();

			int bs_global_columns = bs_storage[0].global.size();
			Eigen::Matrix<Local2Global, Eigen::Dynamic, 1> bs_global_data(basisvalues_size, bs_global_columns);
			Eigen::Matrix<int, Eigen::Dynamic, 1> bs_global_sizes(basisvalues_size, 1);

			Local2Global *bs_global_data_dev = NULL;

			int bsglobaldata_data_size = basisvalues_size * bs_global_columns * sizeof(Local2Global);

			bs_global_data_dev = ALLOCATE_GPU<Local2Global>(bs_global_data_dev, bsglobaldata_data_size);
			cudaDeviceSynchronize();

			for (size_t i = 0; i < basisvalues_size; ++i)
			{
				bs_global_sizes(i, 1) = bs_storage[i].global.size();
				for (int ii = 0; ii < bs_global_columns; ++ii)
				{
					bs_global_data(i, ii) = bs_storage[i].global[ii];
				}

				COPYDATATOGPU<int>(bs_global_sizes_dev, bs_global_sizes.data(), bsglobalsizes_data_size);

				COPYDATATOGPU<Local2Global>(bs_global_data_dev, bs_global_data.data(), bsglobaldata_data_size);

				if (cudaMemcpy(bs_global_data_dev, bs_global_data.data(), bsglobaldata_data_size, cudaMemcpyHostToDevice) != cudaSuccess)
				{
					printf("Error copying to GPU\n");
					abort();
				}

				cudaDeviceSynchronize();

				COPYDATATOHOST<double>(local_dispv, local_dispv_dev, localdispv_data_size);
				cudaDeviceSynchronize();

				DiffScalarBase::setVariableCount(basisvalues_size * size());
				AutoDiffVect local_disp(basisvalues_size * size(), 1);

				Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> def_grad(size(), size());
				int def_grad_size = sizeof(def_grad);
				Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> *def_grad_dev = NULL;

				def_grad_dev = ALLOCATE_GPU<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>>(def_grad_dev, def_grad_size);

				Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> grad;
				Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> *grad_dev;
				int grad_size = sizeof(grad);
				grad_dev = ALLOCATE_GPU<Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>>(grad_dev, grad_size);

				COPYDATATOGPU<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>>(def_grad_dev, &def_grad, def_grad_size);
				for (size_t i = 0; i < vals.basis_values.size(); ++i)
				{
					const auto &bs = vals.basis_values[i];
					assert(grad.size() == size());

					COPYDATATOGPU<Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>>(grad_dev, &grad, grad_size);

					dim3 threads_per_block(NUMBER_THREADS, NUMBER_THREADS);
					defgrad_comp<T><<<1, threads_per_block>>>(def_grad_dev, grad_dev, local_dispv_dev, size(), i);

					cudaDeviceSynchronize();
				}

				COPYDATATOHOST<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>>(&def_grad, def_grad_dev, def_grad_size);

				cudaDeviceSynchronize();
				T energy = 0.0;

				for (long p = 0; p < n_pts; ++p)
				{
					for (long k = 0; k < def_grad.size(); ++k)
						def_grad(k) = T(0);

					for (size_t i = 0; i < vals.basis_values.size(); ++i)
					{
						const auto &bs = vals.basis_values[i];
						const Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> grad = bs.grad.row(p);
						assert(grad.size() == size());

						for (int d = 0; d < size(); ++d)
						{
							for (int c = 0; c < size(); ++c)
							{
								def_grad(d, c) += grad(c) * local_disp(i * size() + d);
							}
						}
					}
					AutoDiffGradMat jac_it(size(), size());
					for (long k = 0; k < jac_it.size(); ++k)
						jac_it(k) = T(vals.jac_it[p](k));
					def_grad = def_grad * jac_it;

					//			COPYDATATOGPU<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>>(jac_it_dev,&jac_it, jac_it_size);

					//Id + grad d
					for (int d = 0; d < size(); ++d)
						def_grad(d, d) += T(1);

					double lambda, mu;
					params_.lambda_mu(vals.quadrature.points.row(p), vals.val.row(p), vals.element_id, lambda, mu);

					const T log_det_j = log(def_grad.determinant());
					const T val = mu / 2 * ((def_grad.transpose() * def_grad).trace() - size() - 2 * log_det_j) + lambda / 2 * log_det_j * log_det_j;

					energy += val * da(p);
				}

				delete[] local_dispv;
				cudaFree(local_dispv_dev);
				cudaFree(def_grad_dev);
				cudaFree(grad_dev);
				cudaFree(displacement_dev);
				cudaFree(bs_global_sizes_dev);
				cudaFree(bs_global_data_dev);
				return energy;
			}
		}
	} // namespace assembler
} // namespace polyfem