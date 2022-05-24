#include <polyfem/NeoHookeanElasticity.hpp>
#include <polyfem/CUDA_utilities.hpp>
#include <polyfem/Basis.hpp>
#include <polyfem/auto_elasticity_rhs.hpp>

#include <polyfem/MatrixUtils.hpp>

namespace polyfem
{

	__global__ void set_dispv(Local2Global* bvs_data, int* bvs_sizes , double* displacement_dev , int size, int bvs_total_size, double* local_disp)
	{
   		int bx = blockIdx.x;
   		int tx = threadIdx.x; 
		int inner_index = bx * NUMBER_THREADS + tx;
		double result=0.0;

		if(inner_index < bvs_total_size)
		{
			for (size_t ii = 0; ii < bvs_sizes[inner_index]; ++ii)
			{
				for (int d = 0; d < size; ++d)
				{
					result= bvs_data[inner_index + ii*bvs_total_size].val * displacement_dev[bvs_data[inner_index+ ii*bvs_total_size].index * size + d];
					local_disp[inner_index * size + d] += result; 
				}
			}
		}
		return;
	}


	double NeoHookeanElasticity::compute_energy(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const 
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

		double *local_dispv_dev=NULL;
		double *displacement_dev = NULL; 
		int *bs_global_sizes_dev = NULL;

		size_t basisvalues_size= vals.basis_values.size();

		int localdispv_data_size= basisvalues_size * size() * sizeof(double); 
		int displ_data_size = displacement.rows()*displacement.cols()*sizeof(double);
		int bsglobalsizes_data_size = basisvalues_size  * sizeof(int); 

		double *local_dispv = new double[basisvalues_size*size()];

		local_dispv_dev = ALLOCATE_GPU<double>(local_dispv_dev,localdispv_data_size);

		displacement_dev = ALLOCATE_GPU<double>(displacement_dev,displ_data_size);

		bs_global_sizes_dev = ALLOCATE_GPU<int>(bs_global_sizes_dev,bsglobalsizes_data_size);

		COPYDATATOGPU<double>(displacement_dev,displacement.data(),displ_data_size);

		cudaDeviceSynchronize();

		const int n_pts = da.size();

//		Eigen::Matrix<double, Eigen::Dynamic, 1> local_dispv(vals.basis_values.size() * size(), 1);
//		local_dispv.setZero();

		const Local2Global *bs_global;

		const AssemblyValues *bs_storage = vals.basis_values.data();
		//const Local2Global **bs_global_data = NULL;

		int bs_global_columns = bs_storage[0].global.size();
		Eigen::Matrix<Local2Global, Eigen::Dynamic,1> bs_global_data(basisvalues_size,bs_global_columns);
		Eigen::Matrix<int, Eigen::Dynamic,1> bs_global_sizes(basisvalues_size,1);

		Local2Global* bs_global_data_dev = NULL;

		int bsglobaldata_data_size = basisvalues_size  * bs_global_columns * sizeof(Local2Global); 

		bs_global_data_dev = ALLOCATE_GPU<Local2Global>(bs_global_data_dev,bsglobaldata_data_size);
		cudaDeviceSynchronize();

		for (size_t i = 0; i < basisvalues_size; ++i)
		{
			bs_global_sizes(i,1) = bs_storage[i].global.size();
			for(int ii=0 ; ii<bs_global_columns;++ii){
				bs_global_data(i,ii) = bs_storage[i].global[ii];
			}
		}	

		COPYDATATOGPU<int>(bs_global_sizes_dev,bs_global_sizes.data(),bsglobalsizes_data_size);

		COPYDATATOGPU<Local2Global>(bs_global_data_dev,bs_global_data.data(),bsglobaldata_data_size);	

		cudaDeviceSynchronize();

		size_t grid_x = (basisvalues_size%NUMBER_THREADS==0) ? basisvalues_size/NUMBER_THREADS : basisvalues_size/NUMBER_THREADS +1;
		set_dispv<<<grid_x,NUMBER_THREADS>>>(bs_global_data_dev, bs_global_sizes_dev, displacement_dev, size(), basisvalues_size, local_dispv_dev);

		cudaDeviceSynchronize();
/*
		for (size_t i = 0; i < basisvalues_size; ++i)
		{
			const auto &bs = vals.basis_values[i];
			for (size_t ii = 0; ii < bs.global.size(); ++ii)
			{
				for (int d = 0; d < size(); ++d)
				{
					local_dispv(i * size() + d) += bs.global[ii].val * displacement(bs.global[ii].index * size() + d);
				}
			}

		}
*/
		COPYDATATOHOST<double>(local_dispv,local_dispv_dev,localdispv_data_size);
		cudaDeviceSynchronize();

		for (size_t i = 0; i < vals.basis_values.size()*size(); ++i)
		{
	//		printf("%lf   ",local_dispv[i]);
		}	

		DiffScalarBase::setVariableCount(basisvalues_size*size());
		AutoDiffVect local_disp(basisvalues_size*size(), 1);

		T energy = T(0.0);

		const AutoDiffAllocator<T> allocate_auto_diff_scalar;

		for (size_t i = 0; i < basisvalues_size*size(); ++i)
		{
			local_disp(i) = allocate_auto_diff_scalar(i, local_dispv[i]);
		}

		AutoDiffGradMat def_grad(size(), size());

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

			//Id + grad d
			for (int d = 0; d < size(); ++d)
				def_grad(d, d) += T(1);

			double lambda, mu;
			params_.lambda_mu(vals.quadrature.points.row(p), vals.val.row(p), vals.element_id, lambda, mu);

			const T log_det_j = log(polyfem::determinant(def_grad));
			const T val = mu / 2 * ((def_grad.transpose() * def_grad).trace() - size() - 2 * log_det_j) + lambda / 2 * log_det_j * log_det_j;

			energy += val * da(p);
		}

		free(local_dispv);
		cudaFree(local_dispv_dev);
		cudaFree(displacement_dev);
		cudaFree(bs_global_sizes_dev);
		cudaFree(bs_global_data_dev);
		return energy;
	}

} // namespace polyfem
