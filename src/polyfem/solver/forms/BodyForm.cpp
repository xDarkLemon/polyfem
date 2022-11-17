#include "BodyForm.hpp"

namespace polyfem::solver
{

#ifdef USE_GPU
	BodyForm::BodyForm(const int ndof,
					   const int n_pressure_bases,
					   const std::vector<int> &boundary_nodes,
					   const std::vector<mesh::LocalBoundary> &local_boundary,
					   const std::vector<mesh::LocalBoundary> &local_neumann_boundary,
					   const int n_boundary_samples,
					   const Eigen::MatrixXd &rhs,
					   const assembler::RhsAssembler &rhs_assembler,
					   const Density &density,
					   const bool apply_DBC,
					   const bool is_formulation_mixed,
					   const bool is_time_dependent,
					   DATA_POINTERS_GPU &data_gpu)
		: ndof_(ndof),
		  n_pressure_bases_(n_pressure_bases),
		  boundary_nodes_(boundary_nodes),
		  local_boundary_(local_boundary),
		  local_neumann_boundary_(local_neumann_boundary),
		  n_boundary_samples_(n_boundary_samples),
		  rhs_(rhs),
		  rhs_assembler_(rhs_assembler),
		  density_(density),
		  apply_DBC_(apply_DBC),
		  is_formulation_mixed_(is_formulation_mixed),
		  data_gpu_(data_gpu)
	{
		t_ = 0;
		if (is_time_dependent)
			update_current_rhs(Eigen::VectorXd());
	}
#endif
#ifndef USE_GPU
	BodyForm::BodyForm(const int ndof,
					   const int n_pressure_bases,
					   const std::vector<int> &boundary_nodes,
					   const std::vector<mesh::LocalBoundary> &local_boundary,
					   const std::vector<mesh::LocalBoundary> &local_neumann_boundary,
					   const int n_boundary_samples,
					   const Eigen::MatrixXd &rhs,
					   const assembler::RhsAssembler &rhs_assembler,
					   const Density &density,
					   const bool apply_DBC,
					   const bool is_formulation_mixed,
					   const bool is_time_dependent)
		: ndof_(ndof),
		  n_pressure_bases_(n_pressure_bases),
		  boundary_nodes_(boundary_nodes),
		  local_boundary_(local_boundary),
		  local_neumann_boundary_(local_neumann_boundary),
		  n_boundary_samples_(n_boundary_samples),
		  rhs_(rhs),
		  rhs_assembler_(rhs_assembler),
		  density_(density),
		  apply_DBC_(apply_DBC),
		  is_formulation_mixed_(is_formulation_mixed)
	{
		t_ = 0;
		if (is_time_dependent)
			update_current_rhs(Eigen::VectorXd());
	}
#endif
	double BodyForm::value_unweighted(const Eigen::VectorXd &x) const
	{
#ifdef USE_GPU
		// return rhs_assembler_.compute_energy(x, local_neumann_boundary_, density_, n_boundary_samples_, t_);
		return rhs_assembler_.compute_energy_GPU(x, local_neumann_boundary_, n_boundary_samples_, t_, data_gpu_);
#endif
#ifndef USE_GPU
		return rhs_assembler_.compute_energy(x, local_neumann_boundary_, density_, n_boundary_samples_, t_);
#endif
	}

	void BodyForm::first_derivative_unweighted(const Eigen::VectorXd &, Eigen::VectorXd &gradv) const
	{
		// REMEMBER -!!!!!
		gradv = -current_rhs_;
	}

	void BodyForm::update_quantities(const double t, const Eigen::VectorXd &x)
	{
		this->t_ = t;
		update_current_rhs(x);
	}

	void BodyForm::update_current_rhs(const Eigen::VectorXd &x)
	{
		rhs_assembler_.compute_energy_grad(
			local_boundary_, boundary_nodes_, density_,
			n_boundary_samples_, local_neumann_boundary_,
			rhs_, t_, current_rhs_);

		if (is_formulation_mixed_ && current_rhs_.size() < ndof_)
		{
			current_rhs_.conservativeResize(
				current_rhs_.rows() + n_pressure_bases_, current_rhs_.cols());
			current_rhs_.bottomRows(n_pressure_bases_).setZero();
		}

		// Apply Neumann boundary conditions
		rhs_assembler_.set_bc(
			std::vector<mesh::LocalBoundary>(), std::vector<int>(),
			n_boundary_samples_, local_neumann_boundary_,
			current_rhs_, x, t_);

		// Apply Dirichlet boundary conditions
		if (apply_DBC_)
			rhs_assembler_.set_bc(
				local_boundary_, boundary_nodes_,
				n_boundary_samples_, std::vector<mesh::LocalBoundary>(),
				current_rhs_, x, t_);
	}
} // namespace polyfem::solver
