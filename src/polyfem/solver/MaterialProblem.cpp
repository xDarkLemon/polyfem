#include "MaterialProblem.hpp"

#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <igl/writeOBJ.h>

#include <filesystem>

namespace polyfem
{
	using namespace utils;

	MaterialProblem::MaterialProblem(State &state_, const std::shared_ptr<CompositeFunctional> j_) : OptimizationProblem(state_, j_)
	{
		optimization_name = "material";

		x_to_param = [](const TVector &x, State &state) {
			auto cur_lambdas = state.assembler.lame_params().lambda_mat_;
			auto cur_mus = state.assembler.lame_params().mu_mat_;

			for (int e = 0; e < cur_mus.size(); e++)
			{
				cur_mus(e) = x(e + cur_mus.size());
				cur_lambdas(e) = x(e);
			}
			state.assembler.update_lame_params(cur_lambdas, cur_mus);
		};

		param_to_x = [](TVector &x, State &state) {
			const auto &cur_lambdas = state.assembler.lame_params().lambda_mat_;
			const auto &cur_mus = state.assembler.lame_params().mu_mat_;

			x.setZero(cur_lambdas.size() + cur_mus.size());
			for (int e = 0; e < cur_mus.size(); e++)
			{
				x(e + cur_mus.size()) = cur_mus(e);
				x(e) = cur_lambdas(e);
			}
		};

		dparam_to_dx = [](TVector &dx, const Eigen::VectorXd &dparams, State &state) {
			dx = dparams.head(state.bases.size() * 2);
		};

		for (const auto &param : opt_params["parameters"])
		{
			if (param["type"] == "material")
			{
				material_params = param;
				break;
			}
		}

		min_mu = 0;
		if (material_params.contains("min_mu"))
			min_mu = material_params["min_mu"];
		max_mu = std::numeric_limits<double>::max();
		if (material_params.contains("max_mu"))
			max_mu = material_params["max_mu"];

		min_lambda = 0;
		if (material_params.contains("min_lambda"))
			min_lambda = material_params["min_lambda"];
		max_lambda = std::numeric_limits<double>::max();
		if (material_params.contains("max_lambda"))
			max_lambda = material_params["max_lambda"];

		has_material_smoothing = false;
		for (const auto &param : opt_params["functionals"])
		{
			if (param["type"] == "material_smoothing")
			{
				smoothing_params = param;
				has_material_smoothing = true;
				smoothing_weight = smoothing_params.value("weight", 1.0);
				break;
			}
		}
		
		// Only works for 2d for now
		if (has_material_smoothing && !state.mesh->is_volume())
		{
			std::vector<Eigen::Triplet<bool>> tt_adjacency_list;
			const mesh::Mesh2D &mesh2d = *dynamic_cast<const mesh::Mesh2D *>(state.mesh.get());
			for (int i = 0; i < state.mesh->n_faces(); ++i)
			{
				auto idx = mesh2d.get_index_from_face(i);
				assert(idx.face == i);
				{
					auto adjacent_idx = mesh2d.switch_face(idx);
					if (adjacent_idx.face != -1)
						tt_adjacency_list.emplace_back(idx.face, adjacent_idx.face, true);
				}
				idx = mesh2d.next_around_face(idx);
				assert(idx.face == i);
				{
					auto adjacent_idx = mesh2d.switch_face(idx);
					if (adjacent_idx.face != -1)
						tt_adjacency_list.emplace_back(idx.face, adjacent_idx.face, true);
				}
				idx = mesh2d.next_around_face(idx);
				assert(idx.face == i);
				{
					auto adjacent_idx = mesh2d.switch_face(idx);
					if (adjacent_idx.face != -1)
						tt_adjacency_list.emplace_back(idx.face, adjacent_idx.face, true);
				}
			}
			tt_adjacency.resize(state.mesh->n_faces(), state.mesh->n_faces());
			tt_adjacency.setFromTriplets(tt_adjacency_list.begin(), tt_adjacency_list.end());
		}
	}

	void MaterialProblem::line_search_end(bool failed)
	{
		if (opt_output_params.contains("export_energies"))
		{
			std::ofstream outfile;
			outfile.open(opt_output_params["export_energies"].get<std::string>(), std::ofstream::out | std::ofstream::app);

			outfile << value(cur_x) << ", " << target_value(cur_x) << ", " << smooth_value(cur_x) << "\n";
			outfile.close();
		}
	}

	double MaterialProblem::max_step_size(const TVector &x0, const TVector &x1)
	{
		double size = 1;
		const auto lambda0 = state.assembler.lame_params().lambda_mat_;
		const auto mu0 = state.assembler.lame_params().mu_mat_;
		while (size > 0)
		{
			auto newX = x0 + (x1 - x0) * size;
			x_to_param(newX, state);
			const auto lambda1 = state.assembler.lame_params().lambda_mat_;
			const auto mu1 = state.assembler.lame_params().mu_mat_;

			const double max_change = opt_nonlinear_params.value("max_change", std::numeric_limits<double>::max());
			if ((lambda1 - lambda0).cwiseAbs().maxCoeff() > max_change || (mu1 - mu0).cwiseAbs().maxCoeff() > max_change || !is_step_valid(x0, newX))
				size /= 2.;
			else
				break;
		}
		x_to_param(x0, state);

		return size;
	}

	double MaterialProblem::target_value(const TVector &x)
	{
		return target_weight * j->energy(state);
	}

	double MaterialProblem::smooth_value(const TVector &x)
	{
		if (!has_material_smoothing || state.mesh->is_volume())
			return 0;

		// no need to use x because x_to_state was called in the solve
		const auto &lambdas = state.assembler.lame_params().lambda_mat_;
		const auto &mus = state.assembler.lame_params().mu_mat_;

		double value = 0;
		for (int k = 0; k < tt_adjacency.outerSize(); ++k)
			for (SparseMatrix<bool>::InnerIterator it(tt_adjacency, k); it; ++it)
			{
				value += pow((1 - lambdas(it.row()) / lambdas(it.col())), 2);
				value += pow((1 - mus(it.row()) / mus(it.col())), 2);
			}
		value /= 3 * tt_adjacency.rows();
		return smoothing_weight * value;
	}

	double MaterialProblem::value(const TVector &x)
	{
		if (std::isnan(cur_val))
		{
			double target_val, smooth_val;
			target_val = target_value(x);
			smooth_val = smooth_value(x);
			logger().debug("target = {}, smooth = {}", target_val, smooth_val);
			cur_val = target_val + smooth_val;
		}
		return cur_val;
	}

	void MaterialProblem::target_gradient(const TVector &x, TVector &gradv)
	{
		Eigen::VectorXd dparam = j->gradient(state, state.problem->is_time_dependent() ? "material-full" : "material");

		dparam_to_dx(gradv, dparam, state);
		gradv *= target_weight;
	}

	void MaterialProblem::smooth_gradient(const TVector &x, TVector &gradv)
	{
		if (!has_material_smoothing || state.mesh->is_volume())
		{
			gradv.setZero(x.size());
			return;
		}

		const auto &lambdas = state.assembler.lame_params().lambda_mat_;
		const auto &mus = state.assembler.lame_params().mu_mat_;
		Eigen::MatrixXd dJ_dmu, dJ_dlambda;
		dJ_dmu.setZero(mus.size(), 1);
		dJ_dlambda.setZero(lambdas.size(), 1);

		for (int k = 0; k < tt_adjacency.outerSize(); ++k)
			for (SparseMatrix<bool>::InnerIterator it(tt_adjacency, k); it; ++it)
			{
				dJ_dlambda(it.row()) += 2 * (lambdas(it.row()) / lambdas(it.col()) - 1) / lambdas(it.col());
				dJ_dlambda(it.col()) += 2 * (1 - lambdas(it.row()) / lambdas(it.col())) * lambdas(it.row()) / lambdas(it.col()) / lambdas(it.col());
				dJ_dmu(it.row()) += 2 * (mus(it.row()) / mus(it.col()) - 1) / mus(it.col());
				dJ_dmu(it.col()) += 2 * (1 - mus(it.row()) / mus(it.col())) * mus(it.row()) / mus(it.col()) / mus(it.col());
			}

		dJ_dmu /= 3 * tt_adjacency.rows();
		dJ_dlambda /= 3 * tt_adjacency.rows();

		Eigen::VectorXd dparam;
		dparam.setZero(dJ_dmu.size() + dJ_dlambda.size() + 3);
		dparam.head(dJ_dlambda.size()) = dJ_dlambda;
		dparam.segment(dJ_dlambda.size(), dJ_dmu.size()) = dJ_dmu;

		dparam_to_dx(gradv, dparam, state);
		gradv *= smoothing_weight;
	}

	void MaterialProblem::gradient(const TVector &x, TVector &gradv)
	{
		if (cur_grad.size() == 0)
		{
			Eigen::VectorXd grad_target, grad_smoothing;
			target_gradient(x, grad_target);
			smooth_gradient(x, grad_smoothing);
			logger().debug("‖∇ target‖ = {}, ‖∇ smooth‖ = {}", grad_target.norm(), grad_smoothing.norm());
			cur_grad = grad_target + grad_smoothing;
		}

		gradv = cur_grad;
	}

	bool MaterialProblem::is_step_valid(const TVector &x0, const TVector &x1)
	{
		const auto &cur_lambdas = state.assembler.lame_params().lambda_mat_;
		const auto &cur_mus = state.assembler.lame_params().mu_mat_;
		const double mu = state.args["contact"]["friction_coefficient"];
		const double psi = state.damping_assembler.local_assembler().get_psi();
		const double phi = state.damping_assembler.local_assembler().get_phi();

		if (mu < 0 || psi < 0 || phi < 0)
			return false;

		if (cur_lambdas.minCoeff() < min_lambda || cur_mus.minCoeff() < min_mu)
			return false;
		if (cur_lambdas.maxCoeff() > max_lambda || cur_mus.maxCoeff() > max_mu)
			return false;
		if (opt_params.contains("min_phi") && opt_params["min_phi"].get<double>() > phi)
			return false;
		if (opt_params.contains("max_phi") && opt_params["max_phi"].get<double>() < phi)
			return false;
		if (opt_params.contains("min_psi") && opt_params["min_psi"].get<double>() > psi)
			return false;
		if (opt_params.contains("max_psi") && opt_params["max_psi"].get<double>() < psi)
			return false;
		if (opt_params.contains("min_fric") && opt_params["min_fric"].get<double>() > mu)
			return false;
		if (opt_params.contains("max_fric") && opt_params["max_fric"].get<double>() < mu)
			return false;

		return true;
	}

	bool MaterialProblem::solution_changed_pre(const TVector &newX)
	{
		x_to_param(newX, state);
		state.set_materials();
		return true;
	}

} // namespace polyfem