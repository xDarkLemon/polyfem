#include <polyfem/Assembler.hpp>
#include <polyfem/CUDA_utilities.cuh>
#include <polyfem/NeoHookeanElasticity.hpp>
#include <polyfem/MultiModel.hpp>
#include "cublas_v2.h"
// #include <polyfem/OgdenElasticity.hpp>

#include <polyfem/Logger.hpp>
//#include <polyfem/MaybeParallelFor.hpp>

#include <igl/Timer.h>

#include <ipc/utils/eigen_ext.hpp>

namespace polyfem
{

	template <class LocalAssembler>
	double NLAssembler<LocalAssembler>::assemble_GPU(
//		const bool is_volume,
//		const std::vector<ElementBases> &bases,
//		const std::vector<ElementBases> &gbases,
		const AssemblyValsCache &cache,
		const Eigen::MatrixXd &displacement) const
	{
		const int n_bases = cache.cache_data_size();
		double store_val = 0.0;

		const ElementAssemblyValues* vals_array = cache.access_cache_data();

		thrust::device_vector<double> displacement_dev(displacement.col(0).begin(),displacement.col(0).end());

		int jac_it_N = vals_array[0].jac_it.size();
		thrust::device_vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>> jac_it_dev(n_bases*jac_it_N);


		int basis_values_N = vals_array[0].basis_values.size();
		int global_columns_N = vals_array[0].basis_values[0].global.size();
		thrust::device_vector<Local2Global> global_data_dev(global_columns_N*basis_values_N);

		thrust::host_vector<Eigen::Matrix<double,-1,1,0,3,1>> da_host(n_bases);

		for (int e = 0; e < n_bases; ++e)
		{
			assert(MAX_QUAD_POINTS == -1 || vals_array[e].quadrature.weights.size() < MAX_QUAD_POINTS);
			int N = vals_array[e].det.size();
			da_host[e].resize(N,1);
			da_host[e] = vals_array[e].det.array() * vals_array[e].quadrature.weights.array();

			thrust::copy(vals_array[e].jac_it.begin(),vals_array[e].jac_it.end(), jac_it_dev.begin()+e*jac_it_N);
			for (int f = 0 ; f<basis_values_N; f++)
				thrust::copy(vals_array[e].basis_values[f].global.begin(),vals_array[e].basis_values[f].global.end(), global_data_dev.begin()+e*(basis_values_N*global_columns_N)+f*global_columns_N);
		}

		thrust::device_vector<Eigen::Matrix<double,-1,1,0,3,1>> da_dev(n_bases);
		thrust::copy(da_host.begin(), da_host.end(), da_dev.begin());

		double lambda, mu;
		const int n_pts = da_host[0].size();

		thrust::device_vector<Eigen::Matrix<double,-1,1,0,3,1>> grad_dev(n_bases*basis_values_N*n_pts);
		for (int e = 0; e < n_bases; ++e)
		{
			for (int f = 0 ; f<basis_values_N; f++){
				for(int p =0; p<n_pts;p++)
					grad_dev[e*basis_values_N*n_pts+f*n_pts+p] = vals_array[e].basis_values[f].grad.row(p);
//					thrust::copy(vals_array[e].basis_values[f].grad.row(p).begin(),vals_array[e].basis_values[f].grad.row(p).end(), grad_dev.begin()+e*(basis_values_N*global_columns_N)+f*global_columns_N+p);
			}
		}


// extract all lambdas and mus and set to device vector
		thrust::device_vector<double> lambda_array(n_pts);
		thrust::device_vector<double> mu_array(n_pts);
		for (int p=0; p<n_pts; p++){

			local_assembler_.get_lambda_mu(vals_array[0].quadrature.points.row(p), vals_array[0].val.row(p),vals_array[0].element_id, lambda, mu);
			lambda_array[p] = lambda;
			mu_array[p] = mu;
		}

// READY TO SEND ALL TO GPU

	double* displacement_dev_ptr = thrust::raw_pointer_cast( displacement_dev.data() );
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>* jac_it_dev_ptr =thrust::raw_pointer_cast( jac_it_dev.data() );
	Local2Global* global_data_dev_ptr = thrust::raw_pointer_cast( global_data_dev.data() );
	Eigen::Matrix<double,-1,1,0,3,1>* da_dev_ptr = thrust::raw_pointer_cast( da_dev.data() );
	Eigen::Matrix<double,-1,1,0,3,1>* grad_dev_ptr = thrust::raw_pointer_cast( grad_dev.data() );

	thrust::device_vector<double> energy_dev_storage(1);
	double* energy_dev_storage_ptr = thrust::raw_pointer_cast(energy_dev_storage.data() );

	local_assembler_.compute_energy_gpu(displacement_dev_ptr,
	jac_it_dev_ptr,
	global_data_dev_ptr,
	da_dev_ptr,
	grad_dev_ptr,
	n_bases,
	basis_values_N,
	global_columns_N,
	energy_dev_storage_ptr
	);


	return store_val;
	}

	//template instantiation
	template class NLAssembler<NeoHookeanElasticity>;
} // namespace polyfem
