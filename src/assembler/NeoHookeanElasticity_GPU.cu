#include <polyfem/NeoHookeanElasticity.hpp>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define NUMBER_THREADS  32

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
/*
		if(inner_index < bvs_total_size)
		{
			for (size_t ii = 0; ii < bvs_sizes(inner_index,1); ++ii)
			{
				for (int d = 0; d < size; ++d)
				{
			//		result= bvs_data(ii,1)->val * displacement(bvs_data(ii,1)->index * size + d);
					result= 0; 
			//		printf("%lf   ",result);
					
					local_disp[inner_index * size + d] += result; 
				}
			}
		}

*/

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

//race_condition
/*		for (int d = 0; d < size; ++d)
		{
			local_disp[inner_index*size + d] += bs_global_val * displacement(bvs_global_index * size + d);
		}
*/
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

		double *local_dispv = new double[vals.basis_values.size()*size()];

		if (cudaMalloc((void **)&local_dispv_dev , localdispv_data_size) != cudaSuccess)
 		{
	      printf("Error allocating to GPU\n");
	      abort();
		}

		cudaDeviceSynchronize();

		if (cudaMalloc((void **)&displacement_dev , displ_data_size) != cudaSuccess)
 		{
	      printf("Error allocating to GPU\n");
	      abort();
		}

		cudaDeviceSynchronize();

		if (cudaMalloc((void **)&bs_global_sizes_dev , bsglobalsizes_data_size) != cudaSuccess)
 		{
	      printf("Error allocating to GPU\n");
	      abort();
		}

		cudaDeviceSynchronize();

	    if(cudaMemcpy(displacement_dev,displacement.data(),displ_data_size,cudaMemcpyHostToDevice) != cudaSuccess)
	    {
		      printf("Error copying to GPU\n");
		      abort(); 
	    }

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
		//int *bs_global_sizes = new int[basisvalues_size];

		Local2Global* bs_global_data_dev = NULL;
		int bsglobaldata_data_size = basisvalues_size  * bs_global_columns* sizeof(Local2Global); 


		if (cudaMalloc((void **)&bs_global_data_dev , bsglobaldata_data_size) != cudaSuccess)
 		{
	      printf("Error allocating to GPU\n");
	      abort();
		}

		cudaDeviceSynchronize();

		for (size_t i = 0; i < basisvalues_size; ++i)
		{
			//bs_global_data[i] = bs_storage[i].global.data();
	//		bs_global_data(i,1) = bs_storage[i].global;
			bs_global_sizes(i,1) = bs_storage[i].global.size();
//			printf("\n%d", bs_storage[i].global.size());
			for(int ii=0 ; ii<bs_global_columns;++ii){
				bs_global_data(i,ii) = bs_storage[i].global[ii];
			}
		}	

		if(cudaMemcpy(bs_global_sizes_dev,bs_global_sizes.data(),bsglobalsizes_data_size,cudaMemcpyHostToDevice) != cudaSuccess)
	    {
		      printf("Error copying to GPU\n");
		      abort(); 
	    }

		cudaDeviceSynchronize();
		
		if(cudaMemcpy(bs_global_data_dev,bs_global_data.data(),bsglobaldata_data_size,cudaMemcpyHostToDevice) != cudaSuccess)
	    {
		      printf("Error copying to GPU\n");
		      abort(); 
	    }

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
		
	    if(cudaMemcpy(local_dispv,local_dispv_dev,localdispv_data_size,cudaMemcpyDeviceToHost) != cudaSuccess)
	    {
		      printf("Error copying to CPU\n");
		      abort(); 
	    }


		cudaDeviceSynchronize();
		for (size_t i = 0; i < vals.basis_values.size()*size(); ++i)
		{
	//		printf("%lf   ",local_dispv[i]);
		}	

//			printf("\n\n");

		DiffScalarBase::setVariableCount(basisvalues_size*size());
		AutoDiffVect local_disp(basisvalues_size*size(), 1);

//		DiffScalarBase::setVariableCount(local_dispv.rows());
//		AutoDiffVect local_disp(local_dispv.rows(), 1);
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
