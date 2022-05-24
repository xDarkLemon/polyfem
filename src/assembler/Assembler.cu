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

		size_t grid_x = (n_bases%NUMBER_THREADS==0) ? n_bases/NUMBER_THREADS : n_bases/NUMBER_THREADS +1;
// This shoud be a kernel CUDA
//		maybe_parallel_for(n_bases, [&](int start, int end, int thread_id) {
			ElementAssemblyValues &vals = storage.vals;
			for (int e = 0; e < n_bases; ++e)
			{
				cache.compute(e, is_volume, bases[e], gbases[e], vals);
				const Quadrature &quadrature = vals.quadrature;

				assert(MAX_QUAD_POINTS == -1 || quadrature.weights.size() < MAX_QUAD_POINTS);
				storage.da = vals.det.array() * quadrature.weights.array();
				
				const double val = local_assembler_.compute_energy(vals, displacement, storage.da);
				storage.val += val;
			}

//		});

//		double res = 0;
		// Serially merge local storages
//		for (const LocalThreadScalarStorage &local_storage : storage)
//			res += local_storage.val;
		return storage.val;
	}

	//template instantiation
	template class NLAssembler<NeoHookeanElasticity>;
} // namespace polyfem
