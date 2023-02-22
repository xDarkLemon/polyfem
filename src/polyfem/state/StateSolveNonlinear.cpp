#include <polyfem/State.hpp>

#include <polyfem/solver/forms/BodyForm.hpp>
#include <polyfem/solver/forms/ContactForm.hpp>
#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/solver/forms/FrictionForm.hpp>
#include <polyfem/solver/forms/InertiaForm.hpp>
#include <polyfem/solver/forms/LaggedRegForm.hpp>
#include <polyfem/solver/forms/ALForm.hpp>
#include <polyfem/solver/forms/RayleighDampingForm.hpp>

#include <polyfem/solver/NonlinearSolver.hpp>
#include <polyfem/solver/LBFGSSolver.hpp>
#include <polyfem/solver/SparseNewtonDescentSolver.hpp>
#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/ALSolver.hpp>
#include <polyfem/solver/SolveData.hpp>
#include <polyfem/io/MshWriter.hpp>
#include <polyfem/io/OBJWriter.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/utils/JSONUtils.hpp>

#include <ipc/ipc.hpp>

namespace polyfem
{
	using namespace mesh;
	using namespace solver;
	using namespace time_integrator;
	using namespace io;
	using namespace utils;

	template <typename ProblemType>
	std::shared_ptr<cppoptlib::NonlinearSolver<ProblemType>> State::make_nl_solver(
		const std::string &linear_solver_type) const
	{
		const std::string name = args["solver"]["nonlinear"]["solver"];
		const double dt = problem->is_time_dependent() ? args["time"]["dt"].get<double>() : 1.0;
		if (name == "newton" || name == "Newton")
		{
			json linear_solver_params = args["solver"]["linear"];
			if (!linear_solver_type.empty())
				linear_solver_params["solver"] = linear_solver_type;
			return std::make_shared<cppoptlib::SparseNewtonDescentSolver<ProblemType>>(
				args["solver"]["nonlinear"], linear_solver_params, dt);
		}
		else if (name == "lbfgs" || name == "LBFGS" || name == "L-BFGS")
		{
			return std::make_shared<cppoptlib::LBFGSSolver<ProblemType>>(args["solver"]["nonlinear"], dt);
		}
		else
		{
			throw std::invalid_argument(fmt::format("invalid nonlinear solver type: {}", name));
		}
	}

	void State::solve_transient_tensor_nonlinear(const int time_steps, const double t0, const double dt, Eigen::MatrixXd &sol)
	{
#ifdef USE_GPU
		printing_GPU_info();
		sending_data_to_GPU();
#endif

		init_nonlinear_tensor_solve(sol, t0 + dt);

		save_timestep(t0, 0, t0, dt, sol, Eigen::MatrixXd()); // no pressure

		for (int t = 1; t <= time_steps; ++t)
		{
			solve_tensor_nonlinear(sol, t);

			{
				POLYFEM_SCOPED_TIMER("Update quantities");

				solve_data.time_integrator->update_quantities(sol);

				solve_data.nl_problem->update_quantities(t0 + (t + 1) * dt, sol);

				solve_data.update_dt();
				solve_data.update_barrier_stiffness(sol);
			}

			save_timestep(t0 + dt * t, t, t0, dt, sol, Eigen::MatrixXd()); // no pressure

			logger().info("{}/{}  t={}", t, time_steps, t0 + dt * t);

			const std::string rest_mesh_path = args["output"]["data"]["rest_mesh"].get<std::string>();
			if (!rest_mesh_path.empty())
			{
				Eigen::MatrixXd V;
				Eigen::MatrixXi F;
				build_mesh_matrices(V, F);
				io::MshWriter::write(
					resolve_output_path(fmt::format(args["output"]["data"]["rest_mesh"], t)),
					V, F, mesh->get_body_ids(), mesh->is_volume(), /*binary=*/true);
			}

			solve_data.time_integrator->save_raw(
				resolve_output_path(fmt::format(args["output"]["data"]["u_path"], t)),
				resolve_output_path(fmt::format(args["output"]["data"]["v_path"], t)),
				resolve_output_path(fmt::format(args["output"]["data"]["a_path"], t)));

			// save restart file
			save_restart_json(t0, dt, t);
		}
	}

#ifdef USE_GPU
	void State::printing_GPU_info()
	{
		size_t free_bytes = 0, total_bytes = 0;
		cudaMemGetInfo(&free_bytes, &total_bytes);
		std::cout << "Mem GPU Free : " << free_bytes << " bytes" << std::endl;
		std::cout << "Mem GPU Total: " << total_bytes << " bytes" << std::endl;
		size_t sizeLimit = 0;
		cudaDeviceGetLimit(&sizeLimit, cudaLimitMallocHeapSize);
		std::cout << "Original device heap sizeLimit: " << sizeLimit << std::endl;
		int n_elements = int(bases.size());
		/*
		std::cout << "SharedMemoryRequired: "
		  << ":" << CALCULATE SIZE!
		  << std::endl;
		*/
		if (n_elements > 5000)
		{
			//	cudaDeviceSetLimit(cudaLimitMallocHeapSize, sizeLimit * 4);
			//	cudaDeviceGetLimit(&sizeLimit, cudaLimitMallocHeapSize);
			//	std::cout << "Current device heap sizeLimit: " << sizeLimit << std::endl;
		}
	}

	void State::sending_data_to_GPU()
	{
		logger().info("Start moving data to GPU");

		auto is_vol_ = mesh->is_volume();

		int n_elements = 0;
		int jac_it_size = 0;
		int n_loc_bases = 0;
		int global_vector_size = 0;
		int n_pts = 0;

		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> *jac_it_dev_ptr = nullptr;
		basis::Local2Global_GPU *global_data_dev_ptr = nullptr;
		Eigen::Matrix<double, -1, 1, 0, 3, 1> *da_dev_ptr = nullptr;
		Eigen::Matrix<double, -1, 1, 0, 3, 1> *val_dev_ptr = nullptr;
		Eigen::Matrix<double, -1, -1, 0, 3, 3> *grad_dev_ptr = nullptr;
		Eigen::Matrix<double, -1, -1, 0, 3, 3> *forces_dev_ptr = nullptr;
		double *lambda_ptr = nullptr;
		double *mu_ptr = nullptr;
		double *rho_ptr = nullptr;

		n_elements = int(bases.size());

		std::vector<polyfem::assembler::ElementAssemblyValues> vals_array(n_elements);

		auto g_bases = geom_bases();

		Eigen::MatrixXd forces_host;
		Eigen::Matrix<double, -1, -1, 0, 3, 3> forces_;

		forces_dev_ptr = ALLOCATE_GPU(forces_dev_ptr, sizeof(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>) * n_elements);

		for (int e = 0; e < n_elements; ++e)
		{
			ass_vals_cache.compute(e, is_vol_, bases[e], g_bases[e], vals_array[e]);
			// to do: check if parameter t = 0 will affect the forces variable;
			(problem.get())->rhs(assembler, formulation(), vals_array[e].val, 0, forces_host);
			forces_ = Eigen::Map<Eigen::Matrix<double, -1, -1, 0, 3, 3>>(forces_host.data(), 3, 3);
			COPYDATATOGPU(forces_dev_ptr + e, &forces_, sizeof(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>));
		}

		logger().info("Finished computing cache for GPU");
		jac_it_size = vals_array[0].jac_it.size();
		n_loc_bases = vals_array[0].basis_values.size();
		global_vector_size = vals_array[0].basis_values[0].global.size();

		jac_it_dev_ptr = ALLOCATE_GPU(jac_it_dev_ptr, sizeof(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>) * n_elements * jac_it_size);

		std::vector<basis::Local2Global_GPU> global_data_host(n_elements * n_loc_bases * global_vector_size);

		global_data_dev_ptr = ALLOCATE_GPU(global_data_dev_ptr, sizeof(basis::Local2Global_GPU) * n_elements * n_loc_bases * global_vector_size);

		std::vector<Eigen::Matrix<double, -1, 1, 0, 3, 1>> da_host(n_elements);

		for (int e = 0; e < n_elements; ++e)
		{
			int N = vals_array[e].det.size();
			da_host[e].resize(N, 1);
			da_host[e] = vals_array[e].det.array() * vals_array[e].quadrature.weights.array();

			COPYDATATOGPU(jac_it_dev_ptr + e * jac_it_size, vals_array[e].jac_it.data(), sizeof(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>) * jac_it_size);
			for (int f = 0; f < n_loc_bases; f++)
			{
				for (int g = 0; g < global_vector_size; g++)
				{
					global_data_host[e * (n_loc_bases * global_vector_size) + f * global_vector_size + g].index = vals_array[e].basis_values[f].global[g].index;
					global_data_host[e * (n_loc_bases * global_vector_size) + f * global_vector_size + g].val = vals_array[e].basis_values[f].global[g].val;
				}
			}
		}
		COPYDATATOGPU(global_data_dev_ptr, global_data_host.data(), sizeof(basis::Local2Global_GPU) * n_elements * n_loc_bases * global_vector_size);

		da_dev_ptr = ALLOCATE_GPU(da_dev_ptr, sizeof(Eigen::Matrix<double, -1, 1, 0, 3, 1>) * n_elements);
		COPYDATATOGPU(da_dev_ptr, da_host.data(), sizeof(Eigen::Matrix<double, -1, 1, 0, 3, 1>) * n_elements);

		n_pts = da_host[0].size();

		grad_dev_ptr = ALLOCATE_GPU(grad_dev_ptr, sizeof(Eigen::Matrix<double, -1, -1, 0, 3, 3>) * n_elements * n_loc_bases);
		val_dev_ptr = ALLOCATE_GPU(val_dev_ptr, sizeof(Eigen::Matrix<double, -1, -1, 0, 3, 1>) * n_elements * n_loc_bases);

		for (int e = 0; e < n_elements; ++e)
		{
			for (int f = 0; f < n_loc_bases; f++)
			{
				Eigen::Matrix<double, -1, -1, 0, 3, 3> row_(Eigen::Map<Eigen::Matrix<double, -1, -1, 0, 3, 3>>(vals_array[e].basis_values[f].grad.data(), 3, 3));
				Eigen::Matrix<double, -1, 1, 0, 3, 1> val_ = vals_array[e].basis_values[f].val;
				COPYDATATOGPU(grad_dev_ptr + e * n_loc_bases + f, &row_, sizeof(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>));
				COPYDATATOGPU(val_dev_ptr + e * n_loc_bases + f, &val_, sizeof(Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>));
			}
		}

		double lambda, mu, rho;
		lambda_ptr = ALLOCATE_GPU(lambda_ptr, sizeof(double) * n_elements * n_pts);
		mu_ptr = ALLOCATE_GPU(mu_ptr, sizeof(double) * n_elements * n_pts);
		rho_ptr = ALLOCATE_GPU(rho_ptr, sizeof(double) * n_elements * n_pts);

		for (int e = 0; e < n_elements; ++e)
		{
			for (int p = 0; p < n_pts; p++)
			{
				// params_tmp_.lambda_mu(vals_array[e].quadrature.points.row(p), vals_array[e].val.row(p), vals_array[e].element_id, lambda, mu);
				assembler.lame_params().lambda_mu(vals_array[e].quadrature.points.row(p), vals_array[e].val.row(p), vals_array[e].element_id, lambda, mu);
				rho = assembler.density()(vals_array[e].quadrature.points.row(p), vals_array[e].val.row(p), vals_array[e].element_id);
				COPYDATATOGPU(lambda_ptr + e * n_pts + p, &lambda, sizeof(double));
				COPYDATATOGPU(mu_ptr + e * n_pts + p, &mu, sizeof(double));
				COPYDATATOGPU(rho_ptr + e * n_pts + p, &rho, sizeof(double));
			}
		}

		logger().info("Finished moving data to GPU");

		gpuErrchk(cudaPeekAtLastError());
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
		data_gpu_.n_elements = n_elements;
		data_gpu_.jac_it_size = jac_it_size;
		data_gpu_.n_loc_bases = n_loc_bases;
		data_gpu_.global_vector_size = global_vector_size;
		data_gpu_.n_pts = n_pts;

		data_gpu_.jac_it_dev_ptr = jac_it_dev_ptr;
		data_gpu_.global_data_dev_ptr = global_data_dev_ptr;
		data_gpu_.da_dev_ptr = da_dev_ptr;
		data_gpu_.val_dev_ptr = val_dev_ptr;
		data_gpu_.grad_dev_ptr = grad_dev_ptr;
		data_gpu_.forces_dev_ptr = forces_dev_ptr;
		data_gpu_.mu_ptr = mu_ptr;
		data_gpu_.lambda_ptr = lambda_ptr;
		data_gpu_.rho_ptr = rho_ptr;

		return;
	}
#endif

	void State::init_nonlinear_tensor_solve(Eigen::MatrixXd &sol, const double t, const bool init_time_integrator)
	{
		assert(!assembler.is_linear(formulation()) || is_contact_enabled()); // non-linear
		assert(!problem->is_scalar());                                       // tensor
		assert(!assembler.is_mixed(formulation()));

		// --------------------------------------------------------------------
		// Check for initial intersections
		if (is_contact_enabled())
		{
			POLYFEM_SCOPED_TIMER("Check for initial intersections");

			const Eigen::MatrixXd displaced = collision_mesh.displace_vertices(
				utils::unflatten(sol, mesh->dimension()));

			if (ipc::has_intersections(collision_mesh, displaced))
			{
				OBJWriter::write(
					resolve_output_path("intersection.obj"), displaced,
					collision_mesh.edges(), collision_mesh.faces());
				log_and_throw_error("Unable to solve, initial solution has intersections!");
			}
		}

		// --------------------------------------------------------------------
		// Initialize time integrator
		if (problem->is_time_dependent())
		{
			if (init_time_integrator)
			{
				POLYFEM_SCOPED_TIMER("Initialize time integrator");
				solve_data.time_integrator = ImplicitTimeIntegrator::construct_time_integrator(args["time"]["integrator"]);

				Eigen::MatrixXd velocity, acceleration;
				initial_velocity(velocity);
				assert(velocity.size() == sol.size());
				initial_acceleration(acceleration);
				assert(acceleration.size() == sol.size());

				const double dt = args["time"]["dt"];
				solve_data.time_integrator->init(sol, velocity, acceleration, dt);
			}
			assert(solve_data.time_integrator != nullptr);
		}
		else
		{
			solve_data.time_integrator = nullptr;
		}

		// --------------------------------------------------------------------
		// Initialize forms

		const std::vector<std::shared_ptr<Form>> forms = solve_data.init_forms(
			// General
			mesh->dimension(), t,
			// Elastic form
			n_bases, bases, geom_bases(), assembler, ass_vals_cache, formulation(),
			// Body form
			n_pressure_bases, boundary_nodes, local_boundary, local_neumann_boundary,
			n_boundary_samples(), rhs, sol,
			// Inertia form
			args["solver"]["ignore_inertia"], mass,
			// Lagged regularization form
			args["solver"]["advanced"]["lagged_regularization_weight"],
			args["solver"]["advanced"]["lagged_regularization_iterations"],
			// Augmented lagrangian form
			obstacle,
			// Contact form
			args["contact"]["enabled"], collision_mesh, args["contact"]["dhat"],
			avg_mass, args["contact"]["use_convergent_formulation"],
			args["solver"]["contact"]["barrier_stiffness"],
			args["solver"]["contact"]["CCD"]["broad_phase"],
			args["solver"]["contact"]["CCD"]["tolerance"],
			args["solver"]["contact"]["CCD"]["max_iterations"],
			// Friction form
			args["contact"]["friction_coefficient"],
			args["contact"]["epsv"],
			args["solver"]["contact"]["friction_iterations"],
			// Rayleigh damping form
			args["solver"]["rayleigh_damping"]
#ifdef USE_GPU
			,
			data_gpu_
#endif
		);

		// --------------------------------------------------------------------
		// Initialize nonlinear problems

		const int ndof = n_bases * mesh->dimension();
		solve_data.nl_problem = std::make_shared<NLProblem>(
			ndof, boundary_nodes, local_boundary, n_boundary_samples(),
			*solve_data.rhs_assembler, t, forms);

		// --------------------------------------------------------------------

		stats.solver_info = json::array();
	}

	void State::solve_tensor_nonlinear(Eigen::MatrixXd &sol, const int t, const bool init_lagging)
	{
		assert(solve_data.nl_problem != nullptr);
		NLProblem &nl_problem = *(solve_data.nl_problem);

		assert(sol.size() == rhs.size());

		if (nl_problem.uses_lagging())
		{
			if (init_lagging)
			{
				POLYFEM_SCOPED_TIMER("Initializing lagging");
				nl_problem.init_lagging(sol); // TODO: this should be u_prev projected
			}
			logger().info("Lagging iteration 1:");
		}

		// ---------------------------------------------------------------------

		// Save the subsolve sequence for debugging
		int subsolve_count = 0;
		save_subsolve(subsolve_count, t, sol, Eigen::MatrixXd()); // no pressure

		// ---------------------------------------------------------------------

		std::shared_ptr<cppoptlib::NonlinearSolver<NLProblem>> nl_solver = make_nl_solver<NLProblem>();

		ALSolver al_solver(
			nl_solver, solve_data.al_form,
			args["solver"]["augmented_lagrangian"]["initial_weight"],
			args["solver"]["augmented_lagrangian"]["scaling"],
			args["solver"]["augmented_lagrangian"]["max_steps"],
			[&](const Eigen::VectorXd &x) {
				this->solve_data.update_barrier_stiffness(sol);
			});

		al_solver.post_subsolve = [&](const double al_weight) {
			json info;
			nl_solver->get_info(info);
			stats.solver_info.push_back(
				{{"type", al_weight > 0 ? "al" : "rc"},
				 {"t", t}, // TODO: null if static?
				 {"info", info}});
			if (al_weight > 0)
				stats.solver_info.back()["weight"] = al_weight;
			save_subsolve(++subsolve_count, t, sol, Eigen::MatrixXd()); // no pressure
		};

		Eigen::MatrixXd prev_sol = sol;
		al_solver.solve(nl_problem, sol, args["solver"]["augmented_lagrangian"]["force"]);

		// ---------------------------------------------------------------------

		// TODO: Make this more general
		const double lagging_tol = args["solver"]["contact"].value("friction_convergence_tol", 1e-2);

		// Lagging loop (start at 1 because we already did an iteration above)
		bool lagging_converged = !nl_problem.uses_lagging();
		for (int lag_i = 1; !lagging_converged; lag_i++)
		{
			Eigen::VectorXd tmp_sol = nl_problem.full_to_reduced(sol);

			// Update the lagging before checking for convergence
			nl_problem.update_lagging(tmp_sol, lag_i);

			// Check if lagging converged
			Eigen::VectorXd grad;
			nl_problem.gradient(tmp_sol, grad);
			const double delta_x_norm = (prev_sol - sol).lpNorm<Eigen::Infinity>();
			logger().debug("Lagging convergence grad_norm={:g} tol={:g} (||Δx||={:g})", grad.norm(), lagging_tol, delta_x_norm);
			if (grad.norm() <= lagging_tol)
			{
				logger().info(
					"Lagging converged in {:d} iteration(s) (grad_norm={:g} tol={:g})",
					lag_i, grad.norm(), lagging_tol);
				lagging_converged = true;
				break;
			}

			if (delta_x_norm <= 1e-12)
			{
				logger().warn(
					"Lagging produced tiny update between iterations {:d} and {:d} (grad_norm={:g} grad_tol={:g} ||Δx||={:g} Δx_tol={:g}); stopping early",
					lag_i - 1, lag_i, grad.norm(), lagging_tol, delta_x_norm, 1e-6);
				lagging_converged = false;
				break;
			}

			// Check for convergence first before checking if we can continue
			if (lag_i >= nl_problem.max_lagging_iterations())
			{
				logger().warn(
					"Lagging failed to converge with {:d} iteration(s) (grad_norm={:g} tol={:g})",
					lag_i, grad.norm(), lagging_tol);
				lagging_converged = false;
				break;
			}

			// Solve the problem with the updated lagging
			logger().info("Lagging iteration {:d}:", lag_i + 1);
			nl_problem.init(sol);
			solve_data.update_barrier_stiffness(sol);
			nl_solver->minimize(nl_problem, tmp_sol);
			prev_sol = sol;
			sol = nl_problem.reduced_to_full(tmp_sol);

			// Save the subsolve sequence for debugging and info
			json info;
			nl_solver->get_info(info);
			stats.solver_info.push_back(
				{{"type", "rc"},
				 {"t", t}, // TODO: null if static?
				 {"lag_i", lag_i},
				 {"info", info}});
			save_subsolve(++subsolve_count, t, sol, Eigen::MatrixXd()); // no pressure
		}
	}

	////////////////////////////////////////////////////////////////////////
	// Template instantiations
	template std::shared_ptr<cppoptlib::NonlinearSolver<NLProblem>> State::make_nl_solver(const std::string &) const;
} // namespace polyfem
