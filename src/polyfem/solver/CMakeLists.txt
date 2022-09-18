set(SOURCES
	NonlinearSolver.hpp
	NonlinearSolver.cu
	SparseNewtonDescentSolver.hpp
	SparseNewtonDescentSolver.cu
	LBFGSSolver.hpp
	NLProblem.cpp
	NLProblem.hpp
	ALNLProblem.cpp
	ALNLProblem.hpp
	NavierStokesSolver.cpp
	NavierStokesSolver.hpp
	TransientNavierStokesSolver.cpp
	TransientNavierStokesSolver.hpp
	OperatorSplittingSolver.hpp
	OperatorSplittingSolver.cpp
)

if(USE_NONLINEAR_GPU)
	add_definitions(-DUSE_NONLINEAR_GPU)
endif()

prepend_current_path(SOURCES)
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})

################################################################################
# Subfolders
################################################################################

add_subdirectory(line_search)
