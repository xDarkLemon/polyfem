set(SOURCES
	ALSolver.cpp
	ALSolver.hpp
	FullNLProblem.cpp
	FullNLProblem.hpp
	LBFGSSolver.hpp
	LBFGSSolver.tpp
	NavierStokesSolver.cpp
	NavierStokesSolver.hpp
	NLProblem.cpp
	NLProblem.hpp
	NonlinearSolver.hpp
	NonlinearSolver.tpp
	OperatorSplittingSolver.hpp
	OperatorSplittingSolver.cpp
	SolveData.cpp
	SolveData.hpp
	SparseNewtonDescentSolver.hpp
	SparseNewtonDescentSolver.tpp
	TransientNavierStokesSolver.cpp
	TransientNavierStokesSolver.hpp
)

if(USE_NONLINEAR_GPU)
	add_definitions(-DUSE_NONLINEAR_GPU)
endif()

prepend_current_path(SOURCES)
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})

# ###############################################################################
# Subfolders
# ###############################################################################
add_subdirectory(line_search)
add_subdirectory(forms)
