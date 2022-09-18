set(SOURCES
	LineSearch.hpp
	LineSearch.tpp
	ArmijoLineSearch.hpp
	BacktrackingLineSearch.hpp
	BacktrackingLineSearch.cu
	CppOptArmijoLineSearch.hpp
	MoreThuenteLineSearch.hpp
)

if(USE_NONLINEAR_GPU)
	add_definitions(-DUSE_NONLINEAR_GPU)
endif()

prepend_current_path(SOURCES)
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
