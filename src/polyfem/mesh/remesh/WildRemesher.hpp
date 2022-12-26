#pragma once

#include <polyfem/mesh/remesh/Remesher.hpp>

#include <wmtk/TriMesh.h>
#include <wmtk/TetMesh.h>
#include <wmtk/ExecutionScheduler.hpp>

namespace polyfem::mesh
{
	template <class WMTKMesh>
	class WildRemesher : public Remesher, public WMTKMesh
	{
		// --------------------------------------------------------------------
		// typedefs
	public:
		// NOTE: This assumes triangle meshes are only used in 2D.
		static constexpr int DIM = [] { if constexpr (std::is_same_v<wmtk::TriMesh, WMTKMesh>) return 2; else return 3; }();

		using Tuple = typename WMTKMesh::Tuple;

		/// @brief Current execuation policy (sequencial or parallel)
		static constexpr wmtk::ExecutionPolicy EXECUTION_POLICY = wmtk::ExecutionPolicy::kSeq;

		// --------------------------------------------------------------------
		// constructors
	public:
		/// @brief Construct a new WildRemesher object
		/// @param state Simulation current state
		WildRemesher(
			const State &state,
			const Eigen::MatrixXd &obstacle_displacements,
			const Eigen::MatrixXd &obstacle_vals,
			const double current_time,
			const double starting_energy);

		virtual ~WildRemesher(){};

		/// @brief Initialize the mesh
		/// @param rest_positions Rest positions of the mesh (|V| × 2)
		/// @param positions Current positions of the mesh (|V| × 2)
		/// @param elements Elements of the mesh (|T| × 3)
		/// @param projection_quantities Quantities to be projected to the new mesh (2 rows per vertex and 1 column per quantity)
		/// @param edge_to_boundary_id Map from edge to boundary id (of size |E|)
		/// @param body_ids Body ids of the mesh (of size |T|)
		virtual void init(
			const Eigen::MatrixXd &rest_positions,
			const Eigen::MatrixXd &positions,
			const Eigen::MatrixXi &elements,
			const Eigen::MatrixXd &projection_quantities,
			const BoundaryMap<int> &boundary_to_id,
			const std::vector<int> &body_ids) override;

		// --------------------------------------------------------------------
		// main functions
	public:
		/// @brief Execute the remeshing
		/// @param split Perform splitting operations
		/// @param collapse Perform collapsing operations
		/// @param smooth Perform smoothing operations
		/// @param swap Perform edge swapping operations
		/// @param max_ops Maximum number of operations to perform (default: unlimited)
		/// @return True if any operation was performed.
		bool execute(
			const bool split = true,
			const bool collapse = false,
			const bool smooth = false,
			const bool swap = false,
			const double max_ops_percent = -1) override;

	protected:
		bool split_edge_before(const Tuple &t) override;

		/// @brief Check if invariants are satisfied
		bool invariants(const std::vector<Tuple> &new_tris) override;

		/// @brief Check if a triangle is inverted
		virtual bool is_inverted(const Tuple &loc) const = 0;

		/// @brief Relax a local n-ring around a vertex.
		/// @param t Center of the local n-ring
		/// @param n_ring Size of the n-ring
		/// @return If the local relaxation reduced the energy "significantly"
		bool local_relaxation(const Tuple &t, const int n_ring);

		// --------------------------------------------------------------------
		// getters
	public:
		/// @brief Dimension of the mesh
		int dim() const override { return DIM; }

		/// @brief Exports rest positions of the stored mesh
		Eigen::MatrixXd rest_positions() const override;
		/// @brief Exports positions of the stored mesh
		Eigen::MatrixXd displacements() const override;
		/// @brief Exports displacements of the stored mesh
		Eigen::MatrixXd positions() const override;
		/// @brief Exports edges of the stored mesh
		Eigen::MatrixXi edges() const override;
		/// @brief Exports elements of the stored mesh
		Eigen::MatrixXi elements() const override;
		/// @brief Exports projected quantities of the stored mesh
		Eigen::MatrixXd projection_quantities() const override;
		/// @brief Exports boundary ids of the stored mesh
		BoundaryMap<int> boundary_ids() const override;
		/// @brief Exports body ids of the stored mesh
		std::vector<int> body_ids() const override;
		/// @brief Get the boundary nodes of the stored mesh
		std::vector<int> boundary_nodes() const override;

		/// @brief Number of projection quantities (not including the position)
		int n_quantities() const override { return m_n_quantities; };

		/// @brief Get a vector of all elements (elements or tetrahedra)
		virtual std::vector<Tuple> get_elements() const = 0;

		// --------------------------------------------------------------------
		// setters
	public:
		/// @brief Set rest positions of the stored mesh
		void set_rest_positions(const Eigen::MatrixXd &positions) override;
		/// @brief Set deformed positions of the stored mesh
		void set_positions(const Eigen::MatrixXd &positions) override;
		/// @brief Set projected quantities of the stored mesh
		void set_projection_quantities(const Eigen::MatrixXd &projection_quantities) override;
		/// @brief Set if a vertex is fixed
		void set_fixed(const std::vector<bool> &fixed) override;
		/// @brief Set the boundary IDs of all edges
		void set_boundary_ids(const BoundaryMap<int> &boundary_to_id) override;
		/// @brief Set the body IDs of all elements
		void set_body_ids(const std::vector<int> &body_ids) override;

		// --------------------------------------------------------------------
		// utilities
	public:
		/// @brief Compute the length of an edge.
		double edge_length(const Tuple &e) const;

		/// @brief Compute the average elastic energy of the faces containing an edge.
		double edge_elastic_energy(const Tuple &e) const;

		/// @brief Compute the volume (area) of an tetrahedron (triangle) element.
		virtual double element_volume(const Tuple &e) const = 0;

		/// @brief Is the given tuple on the boundary of the mesh?
		virtual bool is_on_boundary(const Tuple &t) const = 0;

		/// @brief Get the boundary facets of the mesh
		virtual std::vector<Tuple> boundary_facets() const = 0;

		/// @brief Get the vertex ids of a boundary facet.
		virtual std::array<size_t, DIM> boundary_facet_vids(const Tuple &t) const = 0;

		/// @brief Get the vertex ids of an element.
		virtual std::array<Tuple, DIM + 1> element_vertices(const Tuple &t) const = 0;

		/// @brief Get the vertex ids of an element.
		virtual std::array<size_t, DIM + 1> element_vids(const Tuple &t) const = 0;

		/// @brief Get the one ring of elements around a vertex.
		virtual std::vector<Tuple> get_one_ring_elements_for_vertex(const Tuple &t) const = 0;

		/// @brief Get the id of a facet (edge for triangle, triangle for tetrahedra)
		virtual size_t facet_id(const Tuple &t) const = 0;

		/// @brief Get the id of an element (triangle or tetrahedra)
		virtual size_t element_id(const Tuple &t) const = 0;

		/// @brief Get a tuple of a element with a local facet
		virtual Tuple tuple_from_facet(size_t elem_id, int local_facet_id) const = 0;

		/// @brief Get the incident elements for an edge
		virtual std::vector<Tuple> get_incident_elements_for_edge(const Tuple &t) const = 0;

		/// @brief Create a vector of all the new edge after an operation.
		/// @param tris New elements.
		std::vector<Tuple> new_edges_after(const std::vector<Tuple> &elements) const;

	protected:
		/// @brief Cache the split edge operation
		virtual void cache_split_edge(const Tuple &e) = 0;

		/// @brief Write a visualization mesh of the priority queue
		/// @param e current edge tuple to be split
		void write_priority_queue_mesh(const std::string &path, const Tuple &e);

		// --------------------------------------------------------------------
		// members
	public:
		struct VertexAttributes
		{
			using VectorNd = Eigen::Matrix<double, DIM, 1>;

			VectorNd rest_position;
			VectorNd position;

			/// @brief Quantities to be projected (dim × n_quantities)
			Eigen::MatrixXd projection_quantities;

			bool fixed = false;
			size_t partition_id = 0; // Vertices marked as fixed cannot be modified by any local operation

			VectorNd displacement() const { return position - rest_position; }
		};

		struct BoundaryAttributes
		{
			int boundary_id = -1;
			// TODO: add a field to inidicate if the marked edge was skipped
			// bool skipped = false;
		};

		struct ElementAttributes
		{
			int body_id = 0;
		};

		wmtk::AttributeCollection<VertexAttributes> vertex_attrs;
		wmtk::AttributeCollection<BoundaryAttributes> boundary_attrs;
		wmtk::AttributeCollection<ElementAttributes> element_attrs;

	protected:
		wmtk::ExecutePass<WildRemesher, EXECUTION_POLICY> executor;
		int vis_counter = 0;
		int m_n_quantities;
		double total_volume;
	};

} // namespace polyfem::mesh
