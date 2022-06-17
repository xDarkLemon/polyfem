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

	template<class LocalAssembler>
	__global__ void compute_energy_GPU(ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement , QuadratureVector &da, int n_bases,double *result, LocalAssembler gg)
	{
   		int bx = blockIdx.x;
   		int tx = threadIdx.x; 
		int inner_index = bx * NUMBER_THREADS + tx;
		if(inner_index < n_bases)
		{
		//	double val = gg.compute_energy(vals, displacement, da);
		}

		return;
	}

// ITS MUCH BETTER IDEA TO MOVE ALL ENERGY COMP HERE

	template <class LocalAssembler>
	double NLAssembler<LocalAssembler>::assemble_GPU(
		const bool is_volume,
		const std::vector<ElementBases> &bases,
		const std::vector<ElementBases> &gbases,
		const AssemblyValsCache &cache,
		const Eigen::MatrixXd &displacement) const
	{
		const int n_bases = int(bases.size());

//		double *result_dev=NULL;

//		size_t grid_x = (n_bases%NUMBER_THREADS==0) ? n_bases/NUMBER_THREADS : n_bases/NUMBER_THREADS +1;

		const ElementAssemblyValues* vals_array = cache.access_cache_data();
//		ElementAssemblyValues vals;
		auto da_array = new QuadratureVector[n_bases];
//		QuadratureVector da;
		double store_val = 0.0;


// extract all lambdas and mus
/*
		double lambda, mu;
		const int n_pts = da.size();
		auto lambda_array = new double[n_pts];
		auto mu_array = new double[n_pts];
		for (int p=0; p<n_pts; p++ ){
			local_assembler_.get_lambda_mu(vals.quadrature.points.row(p), vals.val.row(p),vals.element_id, lambda, mu);
			lambda_array[p] = lambda;
			mu_array[p] = mu;
		}
*/
	//	delete [] lambda_array;
	//	delete [] mu_array;
		for (int e = 0; e < n_bases; ++e)
		{
			const Quadrature &quadrature = vals_array[e].quadrature;
			assert(MAX_QUAD_POINTS == -1 || quadrature.weights.size() < MAX_QUAD_POINTS);
			da_array[e] = vals_array[e].det.array() * quadrature.weights.array();
		}

		for (int e = 0; e < n_bases; ++e)
		{
			//cache.compute(e, is_volume, bases[e], gbases[e], vals);
//			const Quadrature &quadrature = vals_array[e].quadrature;

//			assert(MAX_QUAD_POINTS == -1 || quadrature.weights.size() < MAX_QUAD_POINTS);
			//da = vals_array[e].det.array() * quadrature.weights.array();

			const double val = local_assembler_.compute_energy(vals_array[e], displacement, da_array[e]);
			store_val += val;
		}
/*		
		int sumarray_data_size = grid_x * sizeof(double); 
		double *result = new double[sumarray_data_size];

		result_dev = ALLOCATE_GPU<double>(result_dev,sumarray_data_size);

		compute_energy_GPU<LocalAssembler><<<grid_x,NUMBER_THREADS>>>(vals, displacement, da, n_bases, result_dev, local_assembler_);
		//const double val = local_assembler_.compute_energy(vals, displacement, da);
		//store_val += val;

		COPYDATATOHOST<double>(result,result_dev,sumarray_data_size);
		cudaFree(result_dev);
		double test_return = result[0];
		free(result);
		return test_return;
*/

		return store_val;
		
/*
		int sumarray_data_size = grid_x * sizeof(double); 
		double *result = new double[sumarray_data_size];
		result_dev = ALLOCATE_GPU<double>(result_dev,sumarray_data_size);
		COPYDATATOHOST<double>(result,result_dev,sumarray_data_size);
		cudaFree(result_dev);
		double test_return = result[0];
		free(result);
		return test_return;
*/
	}

	//template instantiation
	template class NLAssembler<NeoHookeanElasticity>;
} // namespace polyfem
