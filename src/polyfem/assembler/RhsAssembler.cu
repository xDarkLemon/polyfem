#include "RhsAssembler.hpp"

#include <polyfem/utils/BoundarySampler.hpp>
#include <polysolve/LinearSolver.hpp>

#include <polyfem/utils/Logger.hpp>
#include <igl/Timer.h>
#include <Eigen/Sparse>

#include <iostream>
#include <map>
#include <memory>

namespace polyfem
{
	using namespace polysolve;
	using namespace mesh;
	using namespace quadrature;
	using namespace utils;

	namespace assembler
	{
		using namespace basis;
		__global__ void compute_energy_rhs_GPU(double *displacement,
											   Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 6, 3> *forces_array,
											   Local2Global_GPU *global_data,
											   Eigen::Matrix<double, -1, 1, 0, 6, 1> *da,
											   Eigen::Matrix<double, -1, 1, 0, 6, 1> *val,
											   int n_bases,
											   int n_loc_bases,
											   int global_vector_size,
											   int n_pts,
											   int _size,
											   double *rho_vector,
											   double *energy_val)
		{
			int bx = blockIdx.x;
			int tx = threadIdx.x;
			int b_index = bx * NUMBER_THREADS + tx;

			if (b_index < n_bases)
			{
				Eigen::Matrix<double, -1, 1, 0, 3, 1> local_dispv(_size);
				Eigen::Matrix<double, -1, -1, 0, 6, 3> forces(n_pts, _size);

				double energy = double(0.0);

				for (long p = 0; p < n_pts; ++p)
				{

					local_dispv.setZero();
					for (int i = 0; i < n_loc_bases; ++i)
					{
						const double b_val = val[b_index * n_loc_bases + i](p);
						for (int d = 0; d < _size; ++d)
						{
							for (int ii = 0; ii < global_vector_size; ++ii)
							{
								local_dispv(d) += (global_data[b_index * n_loc_bases * global_vector_size + i * global_vector_size + ii].val * b_val) * displacement[global_data[b_index * n_loc_bases * global_vector_size + i * global_vector_size + ii].index * _size + d];
							}
						}
					}

					const double rho = rho_vector[b_index * n_pts + p];
					forces = forces_array[b_index];
					for (int d = 0; d < _size; ++d)
					{
						///	MAPPING DID A TRANSPOSE
						energy += forces(d, p) * local_dispv(d) * da[b_index](p) * rho;
					}
				}

				atomicAdd(&energy_val[0], energy);
				__syncthreads();
			}
		}

		double RhsAssembler::compute_energy_GPU(const Eigen::MatrixXd &displacement, const std::vector<LocalBoundary> &local_neumann_boundary, const int resolution, const double t, const DATA_POINTERS_GPU &data_gpu) const
		{
			double res = 0;
			igl::Timer timerg;
			if (!problem_.is_rhs_zero())
			{
				const int n_bases = int(bases_.size());
				thrust::device_vector<double> displacement_dev(displacement.col(0).begin(), displacement.col(0).end());
				double *displacement_dev_ptr = thrust::raw_pointer_cast(displacement_dev.data());

				int grid = (n_bases % NUMBER_THREADS == 0) ? n_bases / NUMBER_THREADS : n_bases / NUMBER_THREADS + 1;
				int threads = (n_bases > NUMBER_THREADS) ? NUMBER_THREADS : n_bases;

				thrust::device_vector<double> energy_dev(1, double(0.0));
				double *energy_ptr = thrust::raw_pointer_cast(energy_dev.data());

				compute_energy_rhs_GPU<<<grid, threads>>>(displacement_dev_ptr,
														  data_gpu.forces_dev_ptr,
														  data_gpu.global_data_dev_ptr,
														  data_gpu.da_dev_ptr,
														  data_gpu.val_dev_ptr,
														  data_gpu.n_elements,
														  data_gpu.n_loc_bases,
														  data_gpu.global_vector_size,
														  data_gpu.n_pts,
														  size_,
														  data_gpu.rho_ptr,
														  energy_ptr);

				gpuErrchk(cudaPeekAtLastError());
				CHECK_CUDA_ERROR(cudaDeviceSynchronize());
				thrust::host_vector<double> energy(energy_dev.begin(), energy_dev.end());
				res = energy[0];
			}

			VectorNd local_displacement(size_);
			Eigen::MatrixXd forces;

			ElementAssemblyValues vals;
			// Neumann
			Eigen::MatrixXd points, uv, normals, deform_mat;
			Eigen::VectorXd weights;
			Eigen::VectorXi global_primitive_ids;
			for (const auto &lb : local_neumann_boundary)
			{
				const int e = lb.element_id();
				bool has_samples = utils::BoundarySampler::boundary_quadrature(lb, resolution, mesh_, false, uv, points, normals, weights, global_primitive_ids);

				if (!has_samples)
					continue;

				const basis::ElementBases &gbs = gbases_[e];
				const basis::ElementBases &bs = bases_[e];

				vals.compute(e, mesh_.is_volume(), points, bs, gbs);

				for (int n = 0; n < vals.jac_it.size(); ++n)
				{
					normals.row(n) = normals.row(n) * vals.jac_it[n];
					normals.row(n).normalize();
				}
				problem_.neumann_bc(mesh_, global_primitive_ids, uv, vals.val, normals, t, forces);

				// UIState::ui_state().debug_data().add_points(vals.val, Eigen::RowVector3d(1,0,0));

				for (long p = 0; p < weights.size(); ++p)
				{
					local_displacement.setZero();

					for (size_t i = 0; i < vals.basis_values.size(); ++i)
					{
						const auto &vv = vals.basis_values[i];
						assert(vv.val.size() == weights.size());
						const double b_val = vv.val(p);

						for (int d = 0; d < size_; ++d)
						{
							for (std::size_t ii = 0; ii < vv.global.size(); ++ii)
							{
								local_displacement(d) += (vv.global[ii].val * b_val) * displacement(vv.global[ii].index * size_ + d);
							}
						}
					}

					for (int d = 0; d < size_; ++d)
						res -= forces(p, d) * local_displacement(d) * weights(p);
				}
			}

			return res;
		}
	} // namespace assembler
} // namespace polyfem
