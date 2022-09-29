#include "Assembler.hpp"
#include "CUDA_utilities.cuh"
#include "NeoHookeanElasticity.hpp"
#include "MultiModel.hpp"
#include "cublas_v2.h"
// #include <polyfem/OgdenElasticity.hpp>

#include <polyfem/utils/Logger.hpp>
//#include <polyfem/MaybeParallelFor.hpp>

#include <igl/Timer.h>

#include <ipc/utils/eigen_ext.hpp>

namespace polyfem
{

	using namespace basis;
	using namespace quadrature;
	using namespace utils;

	namespace assembler
	{

		template <class LocalAssembler>
		double NLAssembler<LocalAssembler>::assemble_GPU(
			const bool is_volume,
			const std::vector<ElementBases> &bases,
			const std::vector<ElementBases> &gbases,
			const AssemblyValsCache &cache,
			const Eigen::MatrixXd &displacement) const
		{
			const int n_bases = int(bases.size());
			double store_val = 0.0;

			//const ElementAssemblyValues* vals_array = cache.access_cache_data();
			std::vector<ElementAssemblyValues> vals_array(n_bases);
			for (int e = 0; e < n_bases; ++e)
			{
				cache.compute(e, is_volume, bases[e], gbases[e], vals_array[e]);
			}
			thrust::device_vector<double> displacement_dev(displacement.col(0).begin(), displacement.col(0).end());

			int jac_it_size = vals_array[0].jac_it.size();
			thrust::device_vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>> jac_it_dev(n_bases * jac_it_size);

			int n_loc_bases = vals_array[0].basis_values.size();
			int global_vector_size = vals_array[0].basis_values[0].global.size();
			thrust::device_vector<basis::Local2Global> global_data_dev(n_bases * n_loc_bases * global_vector_size);

			thrust::host_vector<Eigen::Matrix<double, -1, 1, 0, 3, 1>> da_host(n_bases);

			for (int e = 0; e < n_bases; ++e)
			{
				assert(MAX_QUAD_POINTS == -1 || vals_array[e].quadrature.weights.size() < MAX_QUAD_POINTS);
				int N = vals_array[e].det.size();
				da_host[e].resize(N, 1);
				da_host[e] = vals_array[e].det.array() * vals_array[e].quadrature.weights.array();

				thrust::copy(vals_array[e].jac_it.begin(), vals_array[e].jac_it.end(), jac_it_dev.begin() + e * jac_it_size);
				for (int f = 0; f < n_loc_bases; f++)
					thrust::copy(vals_array[e].basis_values[f].global.begin(), vals_array[e].basis_values[f].global.end(), global_data_dev.begin() + e * (n_loc_bases * global_vector_size) + f * global_vector_size);
			}

			thrust::device_vector<Eigen::Matrix<double, -1, 1, 0, 3, 1>> da_dev(n_bases);
			thrust::copy(da_host.begin(), da_host.end(), da_dev.begin());

			double lambda, mu;
			const int n_pts = da_host[0].size();

			thrust::device_vector<Eigen::Matrix<double, -1, 1, 0, 3, 1>> grad_dev(n_bases * n_loc_bases * n_pts);
			for (int e = 0; e < n_bases; ++e)
			{
				for (int f = 0; f < n_loc_bases; f++)
				{
					for (int p = 0; p < n_pts; p++)
						grad_dev[e * n_loc_bases * n_pts + f * n_pts + p] = vals_array[e].basis_values[f].grad.row(p);
					//					thrust::copy(vals_array[e].basis_values[f].grad.row(p).begin(),vals_array[e].basis_values[f].grad.row(p).end(), grad_dev.begin()+e*(n_loc_bases*global_vector_size)+f*global_vector_size+p);
				}
			}

			// extract all lambdas and mus and set to device vector
			thrust::device_vector<double> lambda_array(n_pts);
			thrust::device_vector<double> mu_array(n_pts);
			for (int p = 0; p < n_pts; p++)
			{
				local_assembler_.get_lambda_mu(vals_array[0].quadrature.points.row(p), vals_array[0].val.row(p), vals_array[0].element_id, lambda, mu);
				lambda_array[p] = lambda;
				mu_array[p] = mu;
			}

			// READY TO SEND ALL TO GPU

			double *displacement_dev_ptr = thrust::raw_pointer_cast(displacement_dev.data());
			Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> *jac_it_dev_ptr = thrust::raw_pointer_cast(jac_it_dev.data());
			basis::Local2Global *global_data_dev_ptr = thrust::raw_pointer_cast(global_data_dev.data());
			Eigen::Matrix<double, -1, 1, 0, 3, 1> *da_dev_ptr = thrust::raw_pointer_cast(da_dev.data());
			Eigen::Matrix<double, -1, 1, 0, 3, 1> *grad_dev_ptr = thrust::raw_pointer_cast(grad_dev.data());

			double *lambda_ptr = thrust::raw_pointer_cast(lambda_array.data());
			double *mu_ptr = thrust::raw_pointer_cast(mu_array.data());

			thrust::device_vector<double> energy_dev_storage(n_bases, double(0.0));
			double *energy_dev_storage_ptr = thrust::raw_pointer_cast(energy_dev_storage.data());

			store_val = local_assembler_.compute_energy_gpu(displacement_dev_ptr,
															jac_it_dev_ptr,
															global_data_dev_ptr,
															da_dev_ptr,
															grad_dev_ptr,
															n_bases,
															n_loc_bases,
															global_vector_size,
															n_pts,
															lambda_ptr,
															mu_ptr);

			return store_val;
		}

		template <class LocalAssembler>
		void NLAssembler<LocalAssembler>::assemble_grad_GPU(
			const bool is_volume,
			const int n_basis,
			const std::vector<ElementBases> &bases,
			const std::vector<ElementBases> &gbases,
			const AssemblyValsCache &cache,
			const Eigen::MatrixXd &displacement,
			Eigen::MatrixXd &rhs) const
		{
			rhs.resize(n_basis * local_assembler_.size(), 1);
			rhs.setZero();

			const int n_bases = int(bases.size());

			Eigen::MatrixXd vec;
			vec.resize(rhs.size(), 1);
			vec.setZero();

			std::vector<ElementAssemblyValues> vals_array(n_bases);

			for (int e = 0; e < n_bases; ++e)
			{
				cache.compute(e, is_volume, bases[e], gbases[e], vals_array[e]);
			}
			thrust::device_vector<double> displacement_dev(displacement.col(0).begin(), displacement.col(0).end());
			int jac_it_size = vals_array[0].jac_it.size();

			thrust::device_vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>> jac_it_dev(n_bases * jac_it_size);

			int n_loc_bases = vals_array[0].basis_values.size();
			int global_vector_size = vals_array[0].basis_values[0].global.size();
			thrust::device_vector<basis::Local2Global> global_data_dev(n_bases * n_loc_bases * global_vector_size);

			thrust::host_vector<Eigen::Matrix<double, -1, 1, 0, 3, 1>> da_host(n_bases);

			for (int e = 0; e < n_bases; ++e)
			{
				//assert(MAX_QUAD_POINTS == -1 || quadrature.weights.size() < MAX_QUAD_POINTS);
				int N = vals_array[e].det.size();
				da_host[e].resize(N, 1);
				da_host[e] = vals_array[e].det.array() * vals_array[e].quadrature.weights.array();

				thrust::copy(vals_array[e].jac_it.begin(), vals_array[e].jac_it.end(), jac_it_dev.begin() + e * jac_it_size);
				for (int f = 0; f < n_loc_bases; f++)
				{
					//needs a paranoic check
					thrust::copy(vals_array[e].basis_values[f].global.begin(), vals_array[e].basis_values[f].global.end(), global_data_dev.begin() + e * (n_loc_bases * global_vector_size) + f * global_vector_size);
				}
			}

			thrust::device_vector<Eigen::Matrix<double, -1, 1, 0, 3, 1>> da_dev(n_bases);
			thrust::copy(da_host.begin(), da_host.end(), da_dev.begin());

			const int n_pts = da_host[0].size();

			thrust::device_vector<Eigen::Matrix<double, -1, -1, 0, 3, 3>> grad_dev(n_bases * n_loc_bases * n_pts);
			for (int e = 0; e < n_bases; ++e)
			{
				for (int f = 0; f < n_loc_bases; f++)
				{
					for (int p = 0; p < n_pts; p++)
						grad_dev[e * n_loc_bases * n_pts + f * n_pts + p] = vals_array[e].basis_values[f].grad.row(p);
				}
			}

			// extract all lambdas and mus and set to device vector
			double lambda, mu;
			thrust::device_vector<double> lambda_array(n_pts);
			thrust::device_vector<double> mu_array(n_pts);
			for (int p = 0; p < n_pts; p++)
			{
				local_assembler_.get_lambda_mu(vals_array[0].quadrature.points.row(p), vals_array[0].val.row(p), vals_array[0].element_id, lambda, mu);
				lambda_array[p] = lambda;
				mu_array[p] = mu;
			}

			// READY TO SEND ALL TO GPU

			double *displacement_dev_ptr = thrust::raw_pointer_cast(displacement_dev.data());

			Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> *jac_it_dev_ptr = thrust::raw_pointer_cast(jac_it_dev.data());
			basis::Local2Global *global_data_dev_ptr = thrust::raw_pointer_cast(global_data_dev.data());
			Eigen::Matrix<double, -1, 1, 0, 3, 1> *da_dev_ptr = thrust::raw_pointer_cast(da_dev.data());
			Eigen::Matrix<double, -1, -1, 0, 3, 3> *grad_dev_ptr = thrust::raw_pointer_cast(grad_dev.data());

			double *lambda_ptr = thrust::raw_pointer_cast(lambda_array.data());
			double *mu_ptr = thrust::raw_pointer_cast(mu_array.data());

			vec = local_assembler_.assemble_grad_GPU(displacement_dev_ptr,
													 jac_it_dev_ptr,
													 global_data_dev_ptr,
													 da_dev_ptr,
													 grad_dev_ptr,
													 n_bases,
													 n_loc_bases,
													 global_vector_size,
													 n_pts,
													 lambda_ptr,
													 mu_ptr,
													 n_basis);
			rhs += vec;
			return;
		}

		template <class LocalAssembler>
		void NLAssembler<LocalAssembler>::assemble_hessian_GPU(
			const bool is_volume,
			const int n_basis,
			const bool project_to_psd,
			const std::vector<ElementBases> &bases,
			const std::vector<ElementBases> &gbases,
			const AssemblyValsCache &cache,
			const Eigen::MatrixXd &displacement,
			SpareMatrixCache &mat_cache,
			StiffnessMatrix &grad,
			mapping_pair **mapping) const
		{
			// This is done after calling assemble_hessian to obtain mapping
			igl::Timer timerg;
			mat_cache.init(n_basis * local_assembler_.size());
			mat_cache.set_zero();

			//const int n_bases = int(bases.size());

			//SOME WORK BEING DONE HERE
			static int n_bases = 0;
			static int flag_cache_compute = 0;
			static int jac_it_size = 0;
			static int n_loc_bases = 0;
			static int global_vector_size = 0;
			static int n_pts = 0;

			static Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> *jac_it_dev_ptr = nullptr;
			static basis::Local2Global_GPU *global_data_dev_ptr = nullptr;
			static Eigen::Matrix<double, -1, 1, 0, 3, 1> *da_dev_ptr = nullptr;
			static Eigen::Matrix<double, -1, -1, 0, 3, 3> *grad_dev_ptr = nullptr;

			static double *lambda_ptr = nullptr;
			static double *mu_ptr = nullptr;

			if (!flag_cache_compute)
			{

				timerg.start();

				n_bases = int(bases.size());
				std::vector<ElementAssemblyValues> vals_array(n_bases);

				for (int e = 0; e < n_bases; ++e)
				{
					cache.compute(e, is_volume, bases[e], gbases[e], vals_array[e]);
				}

				jac_it_size = vals_array[0].jac_it.size();
				n_loc_bases = vals_array[0].basis_values.size();
				global_vector_size = vals_array[0].basis_values[0].global.size();

				int check_size_1 = sizeof(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>);
				jac_it_dev_ptr = ALLOCATE_GPU(jac_it_dev_ptr, sizeof(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>) * n_bases * jac_it_size);
				//thrust::device_vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>> jac_it_dev(n_bases * jac_it_size);

				std::vector<basis::Local2Global_GPU> global_data_host(n_bases * n_loc_bases * global_vector_size);

				global_data_dev_ptr = ALLOCATE_GPU(global_data_dev_ptr, sizeof(basis::Local2Global_GPU) * n_bases * n_loc_bases * global_vector_size);
				//thrust::device_vector<basis::Local2Global_GPU> global_data_dev(n_bases * n_loc_bases * global_vector_size);

				std::vector<Eigen::Matrix<double, -1, 1, 0, 3, 1>> da_host(n_bases);

				for (int e = 0; e < n_bases; ++e)
				{
					int N = vals_array[e].det.size();
					da_host[e].resize(N, 1);
					da_host[e] = vals_array[e].det.array() * vals_array[e].quadrature.weights.array();

					//thrust::copy(vals_array[e].jac_it.begin(), vals_array[e].jac_it.end(), jac_it_dev.begin() + e * jac_it_size);
					COPYDATATOGPU(jac_it_dev_ptr + e * jac_it_size, vals_array[e].jac_it.data(), sizeof(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>) * jac_it_size);
					for (int f = 0; f < n_loc_bases; f++)
					{
						for (int g = 0; g < global_vector_size; g++)
						{
							global_data_host[e * (n_loc_bases * global_vector_size) + f * global_vector_size + g].index = vals_array[e].basis_values[f].global[g].index;
							global_data_host[e * (n_loc_bases * global_vector_size) + f * global_vector_size + g].val = vals_array[e].basis_values[f].global[g].val;
						}
					}
				}
				COPYDATATOGPU(global_data_dev_ptr, global_data_host.data(), sizeof(basis::Local2Global_GPU) * n_bases * n_loc_bases * global_vector_size);
				//thrust::copy(global_data_host.begin(), global_data_host.end(), global_data_dev.begin());

				da_dev_ptr = ALLOCATE_GPU(da_dev_ptr, sizeof(Eigen::Matrix<double, -1, 1, 0, 3, 1>) * n_bases);
				//				thrust::device_vector<Eigen::Matrix<double, -1, 1, 0, 3, 1>> da_dev(n_bases);
				COPYDATATOGPU(da_dev_ptr, da_host.data(), sizeof(Eigen::Matrix<double, -1, 1, 0, 3, 1>) * n_bases);
				//				thrust::copy(da_host.begin(), da_host.end(), da_dev.begin());

				n_pts = da_host[0].size();

				//grad_dev_ptr = ALLOCATE_GPU(grad_dev_ptr, sizeof(Eigen::Matrix<double, 1, -1, 0, 1, 3>) * n_bases * n_loc_bases * n_pts);
				grad_dev_ptr = ALLOCATE_GPU(grad_dev_ptr, sizeof(Eigen::Matrix<double, -1, -1, 0, 3, 3>) * n_bases * n_loc_bases * n_pts);
				//thrust::device_vector<Eigen::Matrix<double, -1, -1, 0, 3, 3>> grad_dev(n_bases * n_loc_bases * n_pts);
				for (int e = 0; e < n_bases; ++e)
				{
					for (int f = 0; f < n_loc_bases; f++)
					{
						//						for (int p = 0; p < n_pts; p++)
						//						{
						Eigen::Matrix<double, -1, -1, 0, 3, 3> row_(Eigen::Map<Eigen::Matrix<double, -1, -1, 0, 3, 3>>(vals_array[e].basis_values[f].grad.data(), 3, 3));
						COPYDATATOGPU(grad_dev_ptr + e * n_loc_bases * n_pts + f * n_pts, &row_, sizeof(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>));
						//grad_dev[e * n_loc_bases * n_pts + f * n_pts + p] = vals_array[e].basis_values[f].grad.row(p);
						//						}
					}
				}
				double lambda, mu;
				lambda_ptr = ALLOCATE_GPU(lambda_ptr, sizeof(double) * n_pts);
				mu_ptr = ALLOCATE_GPU(mu_ptr, sizeof(double) * n_pts);

				//				thrust::device_vector<double> lambda_array(n_pts);
				//				thrust::device_vector<double> mu_array(n_pts);
				for (int p = 0; p < n_pts; p++)
				{
					local_assembler_.get_lambda_mu(vals_array[0].quadrature.points.row(p), vals_array[0].val.row(p), vals_array[0].element_id, lambda, mu);
					COPYDATATOGPU(lambda_ptr + p, &lambda, sizeof(double));
					COPYDATATOGPU(mu_ptr + p, &mu, sizeof(double));
					//lambda_array[p] = lambda;
					//mu_array[p] = mu;
				}
				cudaDeviceSynchronize();
				timerg.stop();
				logger().trace("done memory allocations for Assembly Hessian {}s...", timerg.getElapsedTime());
				flag_cache_compute++;
			}
			//SOME WORK BEING DONE HERE

			/*
			timerg.start();
			std::vector<ElementAssemblyValues> vals_array(n_bases);

			for (int e = 0; e < n_bases; ++e)
			{
				cache.compute(e, is_volume, bases[e], gbases[e], vals_array[e]);
			}

			int jac_it_size = vals_array[0].jac_it.size();
			//
			thrust::device_vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>> jac_it_dev(n_bases * jac_it_size);
			//
			int n_loc_bases = vals_array[0].basis_values.size();
			int global_vector_size = vals_array[0].basis_values[0].global.size();
			//

			thrust::host_vector<basis::Local2Global_GPU> global_data_host(n_bases * n_loc_bases * global_vector_size);
			thrust::device_vector<basis::Local2Global_GPU> global_data_dev(n_bases * n_loc_bases * global_vector_size);

			thrust::host_vector<Eigen::Matrix<double, -1, 1, 0, 3, 1>> da_host(n_bases);
			//
			for (int e = 0; e < n_bases; ++e)
			{
				//assert(MAX_QUAD_POINTS == -1 || quadrature.weights.size() < MAX_QUAD_POINTS);
				int N = vals_array[e].det.size();
				da_host[e].resize(N, 1);
				da_host[e] = vals_array[e].det.array() * vals_array[e].quadrature.weights.array();

				thrust::copy(vals_array[e].jac_it.begin(), vals_array[e].jac_it.end(), jac_it_dev.begin() + e * jac_it_size);
				for (int f = 0; f < n_loc_bases; f++)
				{
					//needs a paranoic check
					for (int g = 0; g < global_vector_size; g++)
					{
						global_data_host[e * (n_loc_bases * global_vector_size) + f * global_vector_size + g].index = vals_array[e].basis_values[f].global[g].index;
						global_data_host[e * (n_loc_bases * global_vector_size) + f * global_vector_size + g].val = vals_array[e].basis_values[f].global[g].val;
						//thrust::copy(vals_array[e].basis_values[f].global.begin(), vals_array[e].basis_values[f].global.end(), global_data_dev.begin() + e * (n_loc_bases * global_vector_size) + f * );
					}
				}
			}
			thrust::copy(global_data_host.begin(), global_data_host.end(), global_data_dev.begin());
			//
			thrust::device_vector<Eigen::Matrix<double, -1, 1, 0, 3, 1>> da_dev(n_bases);
			thrust::copy(da_host.begin(), da_host.end(), da_dev.begin());
			//
			const int n_pts = da_host[0].size();
			//
			thrust::device_vector<Eigen::Matrix<double, -1, -1, 0, 3, 3>> grad_dev(n_bases * n_loc_bases * n_pts);
			for (int e = 0; e < n_bases; ++e)
			{
				for (int f = 0; f < n_loc_bases; f++)
				{
					for (int p = 0; p < n_pts; p++)
						grad_dev[e * n_loc_bases * n_pts + f * n_pts + p] = vals_array[e].basis_values[f].grad.row(p);
				}
			}

			// extract all lambdas and mus and set to device vector
			double lambda, mu;
			thrust::device_vector<double> lambda_array(n_pts);
			thrust::device_vector<double> mu_array(n_pts);
			for (int p = 0; p < n_pts; p++)
			{
				local_assembler_.get_lambda_mu(vals_array[0].quadrature.points.row(p), vals_array[0].val.row(p), vals_array[0].element_id, lambda, mu);
				lambda_array[p] = lambda;
				mu_array[p] = mu;
			}

			Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> *jac_it_dev_ptr = thrust::raw_pointer_cast(jac_it_dev.data());
			basis::Local2Global_GPU *global_data_dev_ptr = thrust::raw_pointer_cast(global_data_dev.data());
			//basis::Local2Global *global_data_dev_ptr = nullptr;
			Eigen::Matrix<double, -1, 1, 0, 3, 1> *da_dev_ptr = thrust::raw_pointer_cast(da_dev.data());
			Eigen::Matrix<double, -1, -1, 0, 3, 3> *grad_dev_ptr = thrust::raw_pointer_cast(grad_dev.data());

			double *lambda_ptr = thrust::raw_pointer_cast(lambda_array.data());
			double *mu_ptr = thrust::raw_pointer_cast(mu_array.data());

			timerg.stop();
			logger().trace("done memory allocations for Assembly Hessian {}s...", timerg.getElapsedTime());
			*/

			timerg.start();
			//SENDING DISPLACEMENT TO GPU
			thrust::device_vector<double> displacement_dev(displacement.col(0).begin(), displacement.col(0).end());
			double *displacement_dev_ptr = thrust::raw_pointer_cast(displacement_dev.data());
			//SET UP MOVING VALUES FUNC (MAPPING ALREADY SENT TO GPU)
			std::vector<double> computed_values(mat_cache.non_zeros(), 0);
			timerg.stop();
			logger().trace("done memory allocations for values and transfer displacement to GPU {}s...", timerg.getElapsedTime());

			timerg.start();
			computed_values = local_assembler_.assemble_hessian_GPU(displacement_dev_ptr,
																	jac_it_dev_ptr,
																	global_data_dev_ptr,
																	da_dev_ptr,
																	grad_dev_ptr,
																	n_bases,
																	n_loc_bases,
																	global_vector_size,
																	n_pts,
																	lambda_ptr,
																	mu_ptr,
																	mat_cache.non_zeros(),
																	mapping);
			//computed_values);
			//

			//			// WE NEED TO GO BACK HERE
			//			//if (project_to_psd)
			//			//	stiffness_val = ipc::project_to_psd(stiffness_val);
			//

			//HERE IS THE DEAL
			mat_cache.moving_values(computed_values);
			grad = mat_cache.get_matrix();

			timerg.stop();
			logger().trace("done merge assembly Hessian using GPU {}s...", timerg.getElapsedTime());

			return;
		}

		//template instantiation
		template class NLAssembler<NeoHookeanElasticity>;
	} // namespace assembler
} // namespace polyfem
