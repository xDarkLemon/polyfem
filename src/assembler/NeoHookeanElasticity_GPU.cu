#include <polyfem/NeoHookeanElasticity.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include <polyfem/Basis.hpp>
#include <polyfem/auto_elasticity_rhs.hpp>

#include <polyfem/MatrixUtils.hpp>

namespace polyfem
{

	__global__ void set_dispv(Eigen::Vector3f *v1,double bvs_val, double disp_val, double* local_disp)
	{
		Eigen::MatrixXd lulz(3,1);
		local_disp[0] +=  bvs_val * disp_val;
/*				for (int d = 0; d < size; ++d)
				{
					local_disp[i * size + d] += bs.global[ii].val * displacement(bs.global[ii].index * size + d);
				}
*/
//	double lulz =0.0;
//	res[0] += lulz;
		return;
	}

	double NeoHookeanElasticity::compute_energy_gpu(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) 
	{
		return compute_energy_aux_gpu(vals, displacement, da);
	}

	// Compute ∫ ½μ (tr(FᵀF) - 3 - 2ln(J)) + ½λ ln²(J) du

	double NeoHookeanElasticity::compute_energy_aux_gpu(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) 
	{
		typedef Eigen::Matrix<double, Eigen::Dynamic, 1> AutoDiffVect;
		typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> AutoDiffGradMat;

		assert(displacement.cols() == 1);

		double *local_dispv_dev=NULL;

		int basisvalues_size= vals.basis_values.size();
		int data_size= basisvalues_size * size() * sizeof(double); 

		double *local_dispv = new double[vals.basis_values.size()*size()];

		if (cudaMalloc((void **)&local_dispv_dev , data_size) != cudaSuccess)
 		{
	      printf("Error allocating to GPU\n");
	      abort();
		}

		const int n_pts = da.size();

//		Eigen::Matrix<double, Eigen::Dynamic, 1> local_dispv(vals.basis_values.size() * size(), 1);
//		local_dispv.setZero();

		int size_i = size();
		double bs_global_val,disp_val;

		for (size_t i = 0; i < vals.basis_values.size(); ++i)
		{
			const auto &bs = vals.basis_values[i];
			for (size_t ii = 0; ii < bs.global.size(); ++ii)
			{
				for (int d = 0; d < size(); ++d)
				{
					bs_global_val = bs.global[ii].val;
					disp_val =displacement(bs.global[ii].index * size() + d);
					set_dispv<<<1,1>>>(0,bs_global_val,disp_val,local_dispv_dev);
	//				local_dispv(i * size() + d) += bs.global[ii].val * displacement(bs.global[ii].index * size() + d);
				}
			}
		}

	    if(cudaMemcpy(local_dispv,local_dispv_dev,data_size,cudaMemcpyDeviceToHost) != cudaSuccess)
	    {
		      printf("Error copying to CPU\n");
		      abort(); 
	    }

		DiffScalarBase::setVariableCount(basisvalues_size*size());
		AutoDiffVect local_disp(basisvalues_size*size(), 1);

//		DiffScalarBase::setVariableCount(local_dispv.rows());
//		AutoDiffVect local_disp(local_dispv.rows(), 1);
		double energy = double(0.0);

		const AutoDiffAllocator<double> allocate_auto_diff_scalar;

		for (long i = 0; i < basisvalues_size*size(); ++i)
		{
			local_disp(i) = allocate_auto_diff_scalar(i, local_dispv[i]);
		}

		AutoDiffGradMat def_grad(size(), size());

		for (long p = 0; p < n_pts; ++p)
		{
			for (long k = 0; k < def_grad.size(); ++k)
				def_grad(k) = double(0);

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
				jac_it(k) = double(vals.jac_it[p](k));
			def_grad = def_grad * jac_it;

			//Id + grad d
			for (int d = 0; d < size(); ++d)
				def_grad(d, d) += double(1);

			double lambda, mu;
			params_.lambda_mu(vals.quadrature.points.row(p), vals.val.row(p), vals.element_id, lambda, mu);

			const double log_det_j = log(polyfem::determinant(def_grad));
			const double val = mu / 2 * ((def_grad.transpose() * def_grad).trace() - size() - 2 * log_det_j) + lambda / 2 * log_det_j * log_det_j;

			energy += val * da(p);
		}
		return energy;
	}

} // namespace polyfem
