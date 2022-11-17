#pragma once

#include "AssemblerData.hpp"

#include <polyfem/Common.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>

#ifdef USE_GPU
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include <polyfem/utils/CUDA_utilities.cuh>
#endif

#include <polyfem/assembler/ElementAssemblyValues.hpp>
#include <polyfem/basis/ElementBases.hpp>

#include <polyfem/utils/AutodiffTypes.hpp>
#include <polyfem/utils/Types.hpp>

#include <Eigen/Dense>
#include <array>

// non linear NeoHookean material model
namespace polyfem::assembler
{
	class NeoHookeanElasticity
	{
	public:
		NeoHookeanElasticity();

		// energy, gradient, and hessian used in newton method
		Eigen::MatrixXd assemble_hessian(const NonLinearAssemblerData &data) const;
		Eigen::VectorXd assemble_grad(const NonLinearAssemblerData &data) const;
		double compute_energy(const NonLinearAssemblerData &data) const;
#ifdef USE_GPU
		std::vector<double> assemble_hessian_GPU(double *displacement_dev_ptr,
												 Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> *jac_it_dev_ptr,
												 basis::Local2Global_GPU *global_data_dev_ptr,
												 Eigen::Matrix<double, -1, 1, 0, 3, 1> *da_dev_ptr,
												 Eigen::Matrix<double, -1, -1, 0, 3, 3> *grad_dev_ptr,
												 int n_bases,
												 int n_loc_bases,
												 int global_columns_N,
												 int n_pts,
												 double *lambda_ptr,
												 double *mu_ptr,
												 int non_zeros,
												 //	 mapping_pair **mapping,
												 int **second_cache) const;
		// std::vector<double> &computed_values) const;

		//			template <typename T>
		Eigen::VectorXd assemble_grad_GPU(double *displacement_dev_ptr,
										  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> *jac_it_dev_ptr,
										  basis::Local2Global_GPU *global_data_dev_ptr,
										  Eigen::Matrix<double, -1, 1, 0, 3, 1> *da_dev_ptr,
										  Eigen::Matrix<double, -1, -1, 0, 3, 3> *grad_dev_ptr,
										  int n_bases,
										  int n_loc_bases,
										  int global_columns_N,
										  int n_pts,
										  double *lambda_ptr,
										  double *mu_ptr,
										  //					   double *val_grad_ptr, int n_basis) const;
										  int n_basis) const;

		double compute_energy_gpu(double *displacement_dev_ptr,
								  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> *jac_it_dev_ptr,
								  basis::Local2Global_GPU *global_data_dev_ptr,
								  Eigen::Matrix<double, -1, 1, 0, 3, 1> *da_dev_ptr,
								  Eigen::Matrix<double, -1, -1, 0, 3, 3> *grad_dev_ptr,
								  int n_bases,
								  int n_loc_bases,
								  int global_columns_N,
								  int n_pts,
								  double *lambda_ptr,
								  double *mu_ptr) const;
		//					   double *energy_storage) const;
#endif
		// rhs for fabbricated solution, compute with automatic sympy code
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
		compute_rhs(const AutodiffHessianPt &pt) const;

		inline int size() const { return size_; }
		void set_size(const int size);

		// von mises and stress tensor
		void compute_von_mises_stresses(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const;
		void compute_stress_tensor(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &tensor) const;

		void compute_dstress_dgradu_multiply_mat(const int el_id, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &global_pts, const Eigen::MatrixXd &grad_u_i, const Eigen::MatrixXd &mat, Eigen::MatrixXd &stress, Eigen::MatrixXd &result) const;
		void compute_dstress_dmu_dlambda(const int el_id, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &global_pts, const Eigen::MatrixXd &grad_u_i, Eigen::MatrixXd &dstress_dmu, Eigen::MatrixXd &dstress_dlambda) const;

		// sets material params
		void add_multimaterial(const int index, const json &params);
		void set_params(const LameParameters &params) { params_ = params; }

		// return material params
		void get_lambda_mu(const Eigen::MatrixXd &param, const Eigen::MatrixXd &p, int el_id, double &lambda, double &mu) const;

	private:
		int size_ = -1;

		LameParameters params_;

		// utulity function that computes energy, the template is used for double, DScalar1, and DScalar2 in energy, gradient and hessian
		template <typename T>
		T compute_energy_aux(const NonLinearAssemblerData &data) const;
		template <int n_basis, int dim>
		void compute_energy_hessian_aux_fast(const NonLinearAssemblerData &data, Eigen::MatrixXd &H) const;
		template <int n_basis, int dim>
		void compute_energy_aux_gradient_fast(const NonLinearAssemblerData &data, Eigen::VectorXd &G_flattened) const;

		void assign_stress_tensor(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const int all_size, Eigen::MatrixXd &all, const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const;
	};
} // namespace polyfem::assembler
