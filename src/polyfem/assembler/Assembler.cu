#include "Assembler.hpp"
#include "NeoHookeanElasticity.hpp"
#include "MultiModel.hpp"
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
			const DATA_POINTERS_GPU &data_gpu,
			const Eigen::MatrixXd &displacement) const
		{
			double store_val = 0.0;

			igl::Timer timerg;
			thrust::device_vector<double> displacement_dev(displacement.col(0).begin(), displacement.col(0).end());
			double *displacement_dev_ptr = thrust::raw_pointer_cast(displacement_dev.data());
			timerg.start();
			store_val = local_assembler_.compute_energy_gpu(displacement_dev_ptr,
															data_gpu.jac_it_dev_ptr,
															data_gpu.global_data_dev_ptr,
															data_gpu.da_dev_ptr,
															data_gpu.grad_dev_ptr,
															data_gpu.n_elements,
															data_gpu.n_loc_bases,
															data_gpu.global_vector_size,
															data_gpu.n_pts,
															data_gpu.lambda_ptr,
															data_gpu.mu_ptr);
			timerg.stop();
			logger().trace("done assembly value using GPU {}s...", timerg.getElapsedTime());
			return store_val;
		}

		template <class LocalAssembler>
		void NLAssembler<LocalAssembler>::assemble_grad_GPU(
			const DATA_POINTERS_GPU &data_gpu,
			const int n_basis,
			const Eigen::MatrixXd &displacement,
			Eigen::MatrixXd &rhs) const
		{
			rhs.resize(n_basis * local_assembler_.size(), 1);
			rhs.setZero();

			Eigen::MatrixXd vec;
			vec.resize(rhs.size(), 1);
			vec.setZero();

			igl::Timer timerg;
			thrust::device_vector<double> displacement_dev(displacement.col(0).begin(), displacement.col(0).end());
			double *displacement_dev_ptr = thrust::raw_pointer_cast(displacement_dev.data());
			timerg.start();
			vec = local_assembler_.assemble_grad_GPU(displacement_dev_ptr,
													 data_gpu.jac_it_dev_ptr,
													 data_gpu.global_data_dev_ptr,
													 data_gpu.da_dev_ptr,
													 data_gpu.grad_dev_ptr,
													 data_gpu.n_elements,
													 data_gpu.n_loc_bases,
													 data_gpu.global_vector_size,
													 data_gpu.n_pts,
													 data_gpu.lambda_ptr,
													 data_gpu.mu_ptr,
													 n_basis);
			rhs += vec;
			timerg.stop();
			logger().trace("done assembly gradient using GPU {}s...", timerg.getElapsedTime());
			return;
		}

		template <class LocalAssembler>
		void NLAssembler<LocalAssembler>::assemble_hessian_GPU(
			const DATA_POINTERS_GPU &data_gpu,
			const int n_basis,
			const Eigen::MatrixXd &displacement,
			SpareMatrixCache &mat_cache,
			StiffnessMatrix &grad,
			//			mapping_pair **mapping,
			int **second_cache) const
		{
			// This is done after calling assemble_hessian to obtain mapping

			mat_cache.init(n_basis * local_assembler_.size());
			mat_cache.set_zero();

			igl::Timer timerg;
			timerg.start();
			// SENDING DISPLACEMENT TO GPU
			thrust::device_vector<double> displacement_dev(displacement.col(0).begin(), displacement.col(0).end());
			double *displacement_dev_ptr = thrust::raw_pointer_cast(displacement_dev.data());
			// SET UP MOVING VALUES FUNC (MAPPING ALREADY SENT TO GPU)
			std::vector<double> computed_values(mat_cache.non_zeros(), 0);
			timerg.stop();
			logger().trace("Transfer displacement for Hessian Assembly to GPU {}s...", timerg.getElapsedTime());

			timerg.start();
			computed_values = local_assembler_.assemble_hessian_GPU(displacement_dev_ptr,
																	data_gpu.jac_it_dev_ptr,
																	data_gpu.global_data_dev_ptr,
																	data_gpu.da_dev_ptr,
																	data_gpu.grad_dev_ptr,
																	data_gpu.n_elements,
																	data_gpu.n_loc_bases,
																	data_gpu.global_vector_size,
																	data_gpu.n_pts,
																	data_gpu.lambda_ptr,
																	data_gpu.mu_ptr,
																	mat_cache.non_zeros(),
																	//																	mapping,
																	second_cache);

			//			// WE NEED TO GO BACK HERE
			//			//if (project_to_psd)
			//			//	stiffness_val = ipc::project_to_psd(stiffness_val);
			//

			// HERE IS THE DEAL
			mat_cache.moving_values(computed_values);
			grad = mat_cache.get_matrix();

			timerg.stop();
			logger().trace("done merge assembly Hessian using GPU {}s...", timerg.getElapsedTime());

			return;
		}

		// template instantiation
		template class NLAssembler<NeoHookeanElasticity>;
	} // namespace assembler
} // namespace polyfem
