#include <polyfem/Assembler.hpp>
#include <polyfem/CUDA_utilities.hpp>
#include <polyfem/NeoHookeanElasticity.hpp>
#include <polyfem/MultiModel.hpp>
// #include <polyfem/OgdenElasticity.hpp>

#include <polyfem/Logger.hpp>
//#include <polyfem/MaybeParallelFor.hpp>

#include <igl/Timer.h>

#include <ipc/utils/eigen_ext.hpp>

namespace polyfem
{
	namespace
	{
		class LocalThreadScalarStorage
		{
		public:
			double val;
			ElementAssemblyValues vals;
			QuadratureVector da;

			LocalThreadScalarStorage()
			{
				val = 0;
			}
		};
	} // namespace

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


	template <class LocalAssembler>
	double NLAssembler<LocalAssembler>::assemble_GPU(
		const bool is_volume,
		const std::vector<ElementBases> &bases,
		const std::vector<ElementBases> &gbases,
		const AssemblyValsCache &cache,
		const Eigen::MatrixXd &displacement) const
	{
		auto storage = LocalThreadScalarStorage();
		const int n_bases = int(bases.size());

		double *result_dev=NULL;
		size_t grid_x = (n_bases%NUMBER_THREADS==0) ? n_bases/NUMBER_THREADS : n_bases/NUMBER_THREADS +1;

		ElementAssemblyValues &vals = storage.vals;
		for (int e = 0; e < n_bases; ++e)
		{
			cache.compute(e, is_volume, bases[e], gbases[e], vals);
			const Quadrature &quadrature = vals.quadrature;

			assert(MAX_QUAD_POINTS == -1 || quadrature.weights.size() < MAX_QUAD_POINTS);
			storage.da = vals.det.array() * quadrature.weights.array();
		}

		int sumarray_data_size = grid_x * sizeof(double); 
		double *result = new double[sumarray_data_size];
//		const double val = local_assembler_.compute_energy(vals, displacement, storage.da);
//		storage.val += val;
		result_dev = ALLOCATE_GPU<double>(result_dev,sumarray_data_size);
		compute_energy_GPU<LocalAssembler><<<grid_x,NUMBER_THREADS>>>(vals, displacement, storage.da, n_bases, result_dev, local_assembler_);
		COPYDATATOHOST<double>(result,result_dev,sumarray_data_size);
		cudaFree(result_dev);
		double test_return = result[0];
		free(result);
		return test_return;
	}

	//template instantiation
	template class NLAssembler<NeoHookeanElasticity>;
} // namespace polyfem
