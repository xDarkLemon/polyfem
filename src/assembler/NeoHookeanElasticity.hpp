#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/ElasticityUtils.hpp>

#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include <polyfem/CUDA_utilities.cuh>

#include <polyfem/ElementAssemblyValues.hpp>
#include <polyfem/ElementBases.hpp>
#include <polyfem/AutodiffTypes.hpp>
#include <polyfem/Types.hpp>

#include <Eigen/Dense>
#include <array>

//non linear NeoHookean material model
namespace polyfem
{
	class NeoHookeanElasticity
	{
	public:
		NeoHookeanElasticity();

		//energy, gradient, and hessian used in newton method
		Eigen::MatrixXd assemble_hessian(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const;
		Eigen::VectorXd assemble_grad(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const;

		double compute_energy(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const;

		void compute_energy_gpu(double* displacement_dev_ptr,
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>* jac_it_dev_ptr, 
		Local2Global* global_data_dev_ptr,
		Eigen::Matrix<double,-1,1,0,3,1>* da_dev_ptr,
		Eigen::Matrix<double,-1,1,0,3,1>* grad_dev_ptr,
		int n_bases,
		int basis_values_N,
		int global_columns_N,
		int n_pts,
		double lambda,
		double mu,
		double* energy_storage) const;

		//rhs for fabbricated solution, compute with automatic sympy code
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
		compute_rhs(const AutodiffHessianPt &pt) const;

		inline int size() const { return size_; }
		void set_size(const int size);

		//von mises and stress tensor
		void compute_von_mises_stresses(const int el_id, const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const;
		void compute_stress_tensor(const int el_id, const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &tensor) const;

		void compute_dstress_dgradu_multiply_mat(const int el_id, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &global_pts, const Eigen::MatrixXd &grad_u_i, const Eigen::MatrixXd &mat, Eigen::MatrixXd &stress, Eigen::MatrixXd &result) const;
		void compute_dstress_dmu_dlambda(const int el_id, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &global_pts, const Eigen::MatrixXd &grad_u_i, Eigen::MatrixXd &dstress_dmu, Eigen::MatrixXd &dstress_dlambda) const;

		//sets material params
		void set_parameters(const json &params);
		void init_multimaterial(const bool is_volume, const Eigen::MatrixXd &Es, const Eigen::MatrixXd &nus);
		void set_params(const LameParameters &params) { params_ = params; }
		void get_lambda_mu(const Eigen::MatrixXd &param, const Eigen::MatrixXd &p, int el_id, double &lambda, double &mu) const;

	private:
		int size_ = 2;

		LameParameters params_;

		//utulity function that computes energy, the template is used for double, DScalar1, and DScalar2 in energy, gradient and hessian
		template <typename T>
		T compute_energy_aux(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const;
		template <typename T>
		T compute_energy_aux_deprecated(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const;

		template <int n_basis, int dim>
		void compute_energy_hessian_aux_fast(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da, Eigen::MatrixXd &H) const;
		template <int n_basis, int dim>
		void compute_energy_aux_gradient_fast(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da, Eigen::VectorXd &G_flattened) const;

		void assign_stress_tensor(const int el_id, const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const int all_size, Eigen::MatrixXd &all, const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const;
	};
} // namespace polyfem
