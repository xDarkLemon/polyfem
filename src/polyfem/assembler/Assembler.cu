#include "Assembler.hpp"
#include <assembler/utils/CUDA_utilities.cuh>
#include <polyfem/assembler/NeoHookeanElasticity.hpp>
#include <polyfem/assembler/MultiModel.hpp>
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
		__global__ void compute_energy_GPU(ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, QuadratureVector &da, int n_bases, double *result, LocalAssembler gg)
		{
			int bx = blockIdx.x;
			int tx = threadIdx.x;
			int inner_index = bx * NUMBER_THREADS + tx;
			if (inner_index < n_bases)
			{
				//	double val = gg.compute_energy(vals, displacement, da);
			}
		}

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

			size_t grid_x = (n_bases % NUMBER_THREADS == 0) ? n_bases / NUMBER_THREADS : n_bases / NUMBER_THREADS + 1;

			const ElementAssemblyValues *vals_array = NULL; //TO BE IMPLEMENTED
			auto da_array = new QuadratureVector[n_bases];

			for (int e = 0; e < n_bases; ++e)
			{
				const Quadrature &quadrature = vals_array[e].quadrature;
				assert(MAX_QUAD_POINTS == -1 || vals_array[e].quadrature.weights.size() < MAX_QUAD_POINTS);
				da_array[e] = vals_array[e].det.array() * vals_array[e].quadrature.weights.array();

				// THIS WILL BE A CUDA KERNEL
				const double val = local_assembler_.compute_energy(vals_array[e], displacement, da_array[e]);
				store_val += val;
			}

			return store_val;
		}

		//template instantiation
		template class NLAssembler<NeoHookeanElasticity>;
	} // namespace assembler
} // namespace polyfem