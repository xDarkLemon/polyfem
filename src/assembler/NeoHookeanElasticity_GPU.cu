#include <polyfem/NeoHookeanElasticity.hpp>
#include <polyfem/Basis.hpp>
#include <polyfem/auto_elasticity_rhs.hpp>
#include <cublas_v2.h>
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

	template <typename T> 
	__global__ void defgrad_comp(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>* def_grad, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>* grad, double* local_dispv, int size, int i)
	{
   		int bx = blockIdx.x; int by = blockIdx.y;
   		int tx = threadIdx.x; int ty= threadIdx.y;
		int inner_index_x = bx * NUMBER_THREADS + tx;
		int inner_index_y = by * NUMBER_THREADS + ty;
		if(inner_index_x < size && inner_index_y < size)
		{
				def_grad[0](inner_index_x, inner_index_y) += grad[0](inner_index_y) * local_dispv[i * size + inner_index_x];
		}
		return;
	}

	// Compute ∫ ½μ (tr(FᵀF) - 3 - 2ln(J)) + ½λ ln²(J) du
	template <typename T>
	__global__ void compute_energy_gpu_aux(double* displacement, 
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>* jac_it_array,
	Local2Global* global_data,
	Eigen::Matrix<double,-1,1,0,3,1>* da,
	Eigen::Matrix<double,-1,1,0,3,1>* grad,
	int bv_N,
	int gc_N,
	int general_size,
	T* energy_storage)
	{
		int bx = blockIdx.x;
   		int tx = threadIdx.x; 
		int inner_index = bx * NUMBER_THREADS + tx;

		if(inner_index < 1)
		{
			Eigen::Matrix<double, Eigen::Dynamic, 1> local_dispv(bv_N * general_size, 1);
			local_dispv.setZero();
			for (int i = 0; i < bv_N; ++i)
			{
				for (int ii = 0; ii < gc_N; ++ii)
				{
					for (int d = 0; d < general_size; ++d)
					{
						//take care of the threads it is not complete
						local_dispv(i * general_size + d) += global_data[ii].val * displacement[global_data[ii].index * general_size + d];
					}
				}
			}

			T energy = T(0.0);
			Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> def_grad(general_size, general_size);

			for (long p = 0; p < 1; ++p)
			{
				for (long k = 0; k < def_grad.size(); ++k)
					def_grad(k) = T(0);

				for (size_t i = 0; i < bv_N; ++i)
				{
					for (int d = 0; d < general_size; ++d)
					{
						for (int c = 0; c < general_size; ++c)
						{
							def_grad(d, c) += grad[0](c) * local_dispv(i * general_size + d);
						}
					}
				}

				Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> jac_it(general_size, general_size);
				for (long k = 0; k < jac_it.size(); ++k)
					jac_it(k) = T(jac_it_array[p](k));
				def_grad = def_grad * jac_it;

				//Id + grad d
				for (int d = 0; d < general_size; ++d)
					def_grad(d, d) += T(1);
				double lambda = 1.0;
				double mu = 2.0;
				const T log_det_j = log(def_grad.determinant());
				const T val = mu / 2 * ((def_grad.transpose() * def_grad).trace() - general_size - 2 * log_det_j) + lambda / 2 * log_det_j * log_det_j;

				energy += val * da[inner_index](p);
			}	
		energy_storage[inner_index] = energy;
		}

	}
	

	void NeoHookeanElasticity::compute_energy_gpu(double* displacement_dev_ptr, 
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>* jac_it_dev_ptr,
	Local2Global* global_data_dev_ptr,
	Eigen::Matrix<double,-1,1,0,3,1>* da_dev_ptr,
	Eigen::Matrix<double,-1,1,0,3,1>* grad_dev_ptr,
	int n_bases,
	int bv_N,
	int gc_N,
	double* energy_storage) const
	{
		int grid = (n_bases%NUMBER_THREADS==0) ? n_bases/NUMBER_THREADS : n_bases/NUMBER_THREADS +1;
		int threads = (n_bases>NUMBER_THREADS) ? NUMBER_THREADS : n_bases;
		compute_energy_gpu_aux<double><<<grid,threads>>>(displacement_dev_ptr, jac_it_dev_ptr, global_data_dev_ptr,da_dev_ptr,grad_dev_ptr,bv_N,gc_N,size(),energy_storage);
		return;
	}



	// Compute ∫ ½μ (tr(FᵀF) - 3 - 2ln(J)) + ½λ ln²(J) du
	template <typename T>
	T NeoHookeanElasticity::compute_energy_aux_deprecated(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const
	{

		typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> AutoDiffGradMat;
//		assert(displacement.cols() == 1);

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

		int bs_global_columns = bs_storage[0].global.size();
		Eigen::Matrix<Local2Global, Eigen::Dynamic,Eigen::Dynamic> bs_global_data(basisvalues_size,bs_global_columns);
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
		CATCHCUDAERROR();
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

//		DiffScalarBase::setVariableCount(basisvalues_size*size());
//		AutoDiffVect local_disp(basisvalues_size*size(), 1);

		T energy = T(0.0);

//		const AutoDiffAllocator<T> allocate_auto_diff_scalar;

//		for (size_t i = 0; i < basi//svalues_size*size(); ++i)
//		{
//			local_disp(i) = allocate_auto_diff_scalar(i, local_dispv[i]);
//		}

		Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> def_grad(size(), size());
		int def_grad_size = sizeof(def_grad);
		Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>* def_grad_dev=NULL;

		def_grad_dev = ALLOCATE_GPU<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>>(def_grad_dev,def_grad_size);

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> grad;
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> *grad_dev;
		int grad_size = sizeof(grad);
		grad_dev = ALLOCATE_GPU<Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>>(grad_dev,grad_size);

		for (long p = 0; p < n_pts; ++p)
		{
			for (long k = 0; k < def_grad.size(); ++k)
				def_grad(k) = T(0);

			COPYDATATOGPU<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>>(def_grad_dev,&def_grad,def_grad_size);
			for (size_t i = 0; i < vals.basis_values.size(); ++i)
			{
				const auto &bs = vals.basis_values[i];
				grad = bs.grad.row(p);
				assert(grad.size() == size());

				COPYDATATOGPU<Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>>(grad_dev,&grad,grad_size);

				dim3 threads_per_block(NUMBER_THREADS, NUMBER_THREADS);
				dim3 blocks_per_dimension(grid_x, grid_x);
				defgrad_comp<T><<<blocks_per_dimension,threads_per_block>>>(def_grad_dev,grad_dev,local_dispv_dev,size(), i);

				CATCHCUDAERROR();
				cudaDeviceSynchronize();
/*
				for (int d = 0; d < size(); ++d)
				{
					for (int c = 0; c < size(); ++c)
					{
						def_grad(d, c) += grad(c) * local_dispv[i * size() + d];
					}
				}
*/
			}

			COPYDATATOHOST<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>>(&def_grad,def_grad_dev,def_grad_size);

			cudaDeviceSynchronize();

//			int n_jac_it = vals.jac_it.size();
//			int jac_it_size = sizeof(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>);
//			Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>* jac_it_dev;

//			jac_it_dev = ALLOCATE_GPU<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>>(jac_it_dev,jac_it_size);

//			COPYDATATOGPU<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>>(jac_it_dev,vals.jac_it.data(),jac_it_size*n_jac_it);


			AutoDiffGradMat jac_it(size(), size());
			for (long k = 0; k < jac_it.size(); ++k)
				jac_it(k) = T(vals.jac_it[p](k));
			def_grad = def_grad * jac_it;

//			COPYDATATOGPU<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>>(jac_it_dev,&jac_it, jac_it_size);

			//Id + grad d
			for (int d = 0; d < size(); ++d)
				def_grad(d, d) += T(1);

//			test_matmul<<<1,1>>>();
//			COPYDATATOHOST<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor,2,2>>(&C_h,dev_C,sizeof(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor,2,2>));
//			cudaDeviceSynchronize();
//			CATCHCUDAERROR();


//			COPYDATATOGPU<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>>(def_grad_dev,&def_grad,def_grad_size);

			T* h_ptr_defgrad[1];
			auto defgrad_h_out = new T[9];
			T* d_defgrad;
			T** d_ptr_defgrad;
			auto h_info = new int[1];
			auto result = new double[1];
			double* d_result ;
			int* d_info;

			d_defgrad = ALLOCATE_GPU<T>(d_defgrad,9*sizeof(T));
			d_info = ALLOCATE_GPU<int>(d_info,1*sizeof(int));
			d_ptr_defgrad = ALLOCATE_GPU<T*>(d_ptr_defgrad,1*sizeof(T*));

		    cublasHandle_t handle;
		    cublasCreate(&handle);



			COPYDATATOGPU<T>(d_defgrad,def_grad.data(),9*sizeof(T));


			h_ptr_defgrad[0] = d_defgrad;
			COPYDATATOGPU<T*>(d_ptr_defgrad,h_ptr_defgrad,sizeof(T*));
			cublasDgetrfBatched(handle, 3, d_ptr_defgrad, 3, NULL, d_info,1);

			cublasDestroy(handle);


			cudaDeviceSynchronize();
			COPYDATATOHOST<T>(defgrad_h_out,d_defgrad,9*sizeof(T));
			COPYDATATOHOST<int>(h_info,d_info,1*sizeof(int));

			d_result = ALLOCATE_GPU<double>(d_result,1*sizeof(double));

			mult_trace<double>(d_defgrad, 3, d_result);
			//multiplicative_trace<T><<<1,1>>>(d_defgrad, 3, d_result);
			cudaDeviceSynchronize();

			COPYDATATOHOST<double>(result,d_result,1*sizeof(result));

			double lambda, mu;
			params_.lambda_mu(vals.quadrature.points.row(p), vals.val.row(p), vals.element_id, lambda, mu);


//			printf("%lf   test def_Grad\n",def_grad.determinant());
			const T log_det_j = result[0];
//			const T log_det_j = log(def_grad.determinant());
			const T val = mu / 2 * ((def_grad.transpose() * def_grad).trace() - size() - 2 * log_det_j) + lambda / 2 * log_det_j * log_det_j;

			energy += val * da(p);
			cudaFree(d_defgrad);
			cudaFree(d_info);
			cudaFree(d_result);
			cudaFree(d_ptr_defgrad);


			delete [] h_info;
			delete [] result;


		}
		
		delete [] local_dispv;
		cudaFree(local_dispv_dev);
		cudaFree(def_grad_dev);
		cudaFree(grad_dev);
		cudaFree(displacement_dev);
		cudaFree(bs_global_sizes_dev);
		cudaFree(bs_global_data_dev);
		return energy;	
	}

} // namespace polyfem
