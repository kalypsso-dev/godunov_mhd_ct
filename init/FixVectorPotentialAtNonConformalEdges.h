// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file FixVectorPotentialAtNonConformalEdges.h
 */
#ifndef KALYPSSO_GODUNOV_MHD_CT_INIT_FIX_VECTOR_POTENTIAL_AT_NON_CONFORMAL_EDGES_H_
#define KALYPSSO_GODUNOV_MHD_CT_INIT_FIX_VECTOR_POTENTIAL_AT_NON_CONFORMAL_EDGES_H_
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for EdgeDataArrayBlock

#include <kalypsso/core/ConformalFullStatus.h>
#include <kalypsso/core/utils_block.h>
#include <kalypsso/core/mesh_utils.h>

namespace kalypsso
{

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * A Kokkos functor that modifies an EdgeDataArrayBlock (representing vector potential on AMR grid)
 * in order to fix it at locations of non-conforming edges.
 *
 * When a edge is non-corforming, in order to compute consistently a magnetic field (curl of vector
 * potential), the circulation computed on all "sides" of an edge must be the same.
 * Thus the fix that is applied here consists in replacing the vector potential on the coarse "side"
 * by the average of the vector potential on the fine "side".
 *
 * Actually, to make things simpler, we just recompute the vector potential at those location using
 * a field known analytcially.
 */
template <size_t dim, typename device_t, typename AnalyticalVecPot>
class FixVectorPotentialAtNonConformalEdges
{

public:
  using exec_space = typename device_t::execution_space;
  using index_t = int32_t;
  //! type alias for a (device) Kokkos view of orchard keys
  using orchard_key_view_t = typename MeshMap<dim, device_t>::orchard_key_view_t;

  using EdgeDataArrayBlock_t = EdgeDataArrayBlock<dim, real_t, device_t>;
  // using conformal_full_status_view_type = conformal_full_status_view_t<dim, device_t>;

private:
  //! list of orchard key of the mesh
  orchard_key_view_t m_orchard_keys;

  //! discrete vector potential components
  EdgeDataArrayBlock_t m_vector_potential;

  //! analytical vector potential functor.
  //! class AnalyticalVecPot must be a function with signature
  //! operator()(Kokkos::Array<double, dim> xyz, int ivar) to evaluate vector potential components
  //! at a given space (x,y,z) location.
  AnalyticalVecPot m_ana_vector_potential;

  //! conformal_full_status_view is a view, one integer per AMR leaf
  //! encoding conformal status.
  conformal_full_status_view_t<dim, device_t> m_conformal_full_status_view;

  //! block sizes
  block_size_t<dim> m_block_sizes;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

public:
  // ====================================================================
  // ====================================================================
  FixVectorPotentialAtNonConformalEdges(
    ConfigMap const &                           config_map,
    orchard_key_view_t                          orchard_keys,
    EdgeDataArrayBlock_t                        vector_potential,
    AnalyticalVecPot                            ana_vector_potential,
    conformal_full_status_view_t<dim, device_t> conformal_full_status_view)
    : m_orchard_keys(orchard_keys)
    , m_vector_potential(vector_potential)
    , m_ana_vector_potential(ana_vector_potential)
    , m_conformal_full_status_view(conformal_full_status_view)
    , m_block_sizes(vector_potential.cell_block_size())
    , m_scaling_factor(get_scaling_factor(config_map))
    , m_xyz_min(get_xyz_min<dim>(config_map))
  {}

  // ====================================================================
  // ====================================================================
  static void
  apply(ConfigMap const &                           config_map,
        orchard_key_view_t                          orchard_keys,
        EdgeDataArrayBlock_t                        vector_potential,
        AnalyticalVecPot                            ana_vector_potential,
        conformal_full_status_view_t<dim, device_t> conformal_full_status_view,
        int32_t                                     local_num_octants)
  {
    FixVectorPotentialAtNonConformalEdges functor(
      config_map, orchard_keys, vector_potential, ana_vector_potential, conformal_full_status_view);

    const auto nbEdgesPerLeaf = vector_potential.num_elements_per_octant();
    const auto nbEdgesTotal = local_num_octants * nbEdgesPerLeaf;

    Kokkos::parallel_for("FixVectorPotentialAtNonConformalEdges",
                         Kokkos::RangePolicy<exec_space>(0, nbEdgesTotal),
                         functor);
  } // apply

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION bool
  is_neighbor_finer(edge_multiindex_t<dim> const &             edge_indexes,
                    conformal_full_status_value_t<dim> const & status) const
  {
    if constexpr (dim == 2)
    {

      // in 2d only components along X or Y might need to be modified

      auto const & ivar = edge_indexes[dim];

      if (ivar == IX and is_edge_at_block_surface<dim>(edge_indexes, m_block_sizes, Face::YMIN) and
          conformal_full_status_t<dim>::get_status(Face::YMIN, status) ==
            conformal_neighbor_status::NEIGHBOR_IS_FINER)
      {
        return true;
      }
      else if (ivar == IX and
               is_edge_at_block_surface<dim>(edge_indexes, m_block_sizes, Face::YMAX) and
               conformal_full_status_t<dim>::get_status(Face::YMAX, status) ==
                 conformal_neighbor_status::NEIGHBOR_IS_FINER)
      {
        return true;
      }
      else if (ivar == IY and
               is_edge_at_block_surface<dim>(edge_indexes, m_block_sizes, Face::XMIN) and
               conformal_full_status_t<dim>::get_status(Face::XMIN, status) ==
                 conformal_neighbor_status::NEIGHBOR_IS_FINER)
      {
        return true;
      }
      else if (ivar == IY and
               is_edge_at_block_surface<dim>(edge_indexes, m_block_sizes, Face::XMAX) and
               conformal_full_status_t<dim>::get_status(Face::XMAX, status) ==
                 conformal_neighbor_status::NEIGHBOR_IS_FINER)
      {
        return true;
      }
      return false;
    }
    else if constexpr (dim == 3)
    {
      auto const & ivar = edge_indexes[dim];

      // X edges
      if (ivar == IX and
          is_edge_at_block_edge<dim>(edge_indexes, m_block_sizes, Face::YMIN, Face::ZMIN) and
          conformal_full_status_t<dim>::get_status(Face::YMIN, Face::ZMIN, status) ==
            conformal_neighbor_status::NEIGHBOR_IS_FINER)
      {
        return true;
      }
      else if (ivar == IX and
               is_edge_at_block_edge<dim>(edge_indexes, m_block_sizes, Face::YMAX, Face::ZMIN) and
               conformal_full_status_t<dim>::get_status(Face::YMAX, Face::ZMIN, status) ==
                 conformal_neighbor_status::NEIGHBOR_IS_FINER)
      {
        return true;
      }
      else if (ivar == IX and
               is_edge_at_block_edge<dim>(edge_indexes, m_block_sizes, Face::YMIN, Face::ZMAX) and
               conformal_full_status_t<dim>::get_status(Face::YMIN, Face::ZMAX, status) ==
                 conformal_neighbor_status::NEIGHBOR_IS_FINER)
      {
        return true;
      }
      else if (ivar == IX and
               is_edge_at_block_edge<dim>(edge_indexes, m_block_sizes, Face::YMAX, Face::ZMAX) and
               conformal_full_status_t<dim>::get_status(Face::YMAX, Face::ZMAX, status) ==
                 conformal_neighbor_status::NEIGHBOR_IS_FINER)
      {
        return true;
      }
      else if (ivar == IX and
               is_edge_at_block_border<dim>(edge_indexes, m_block_sizes, Face::YMIN) and
               conformal_full_status_t<dim>::get_status(Face::YMIN, status) ==
                 conformal_neighbor_status::NEIGHBOR_IS_FINER)
      {
        return true;
      }
      else if (ivar == IX and
               is_edge_at_block_border<dim>(edge_indexes, m_block_sizes, Face::YMAX) and
               conformal_full_status_t<dim>::get_status(Face::YMAX, status) ==
                 conformal_neighbor_status::NEIGHBOR_IS_FINER)
      {
        return true;
      }
      else if (ivar == IX and
               is_edge_at_block_border<dim>(edge_indexes, m_block_sizes, Face::ZMIN) and
               conformal_full_status_t<dim>::get_status(Face::ZMIN, status) ==
                 conformal_neighbor_status::NEIGHBOR_IS_FINER)
      {
        return true;
      }
      else if (ivar == IX and
               is_edge_at_block_border<dim>(edge_indexes, m_block_sizes, Face::ZMAX) and
               conformal_full_status_t<dim>::get_status(Face::ZMAX, status) ==
                 conformal_neighbor_status::NEIGHBOR_IS_FINER)
      {
        return true;
      }

      // Y edges
      if (ivar == IY and
          is_edge_at_block_edge<dim>(edge_indexes, m_block_sizes, Face::XMIN, Face::ZMIN) and
          conformal_full_status_t<dim>::get_status(Face::XMIN, Face::ZMIN, status) ==
            conformal_neighbor_status::NEIGHBOR_IS_FINER)
      {
        return true;
      }
      else if (ivar == IY and
               is_edge_at_block_edge<dim>(edge_indexes, m_block_sizes, Face::XMAX, Face::ZMIN) and
               conformal_full_status_t<dim>::get_status(Face::XMAX, Face::ZMIN, status) ==
                 conformal_neighbor_status::NEIGHBOR_IS_FINER)
      {
        return true;
      }
      else if (ivar == IY and
               is_edge_at_block_edge<dim>(edge_indexes, m_block_sizes, Face::XMIN, Face::ZMAX) and
               conformal_full_status_t<dim>::get_status(Face::XMIN, Face::ZMAX, status) ==
                 conformal_neighbor_status::NEIGHBOR_IS_FINER)
      {
        return true;
      }
      else if (ivar == IY and
               is_edge_at_block_edge<dim>(edge_indexes, m_block_sizes, Face::XMAX, Face::ZMAX) and
               conformal_full_status_t<dim>::get_status(Face::XMAX, Face::ZMAX, status) ==
                 conformal_neighbor_status::NEIGHBOR_IS_FINER)
      {
        return true;
      }
      else if (ivar == IY and
               is_edge_at_block_border<dim>(edge_indexes, m_block_sizes, Face::XMIN) and
               conformal_full_status_t<dim>::get_status(Face::XMIN, status) ==
                 conformal_neighbor_status::NEIGHBOR_IS_FINER)
      {
        return true;
      }
      else if (ivar == IY and
               is_edge_at_block_border<dim>(edge_indexes, m_block_sizes, Face::XMAX) and
               conformal_full_status_t<dim>::get_status(Face::XMAX, status) ==
                 conformal_neighbor_status::NEIGHBOR_IS_FINER)
      {
        return true;
      }
      else if (ivar == IY and
               is_edge_at_block_border<dim>(edge_indexes, m_block_sizes, Face::ZMIN) and
               conformal_full_status_t<dim>::get_status(Face::ZMIN, status) ==
                 conformal_neighbor_status::NEIGHBOR_IS_FINER)
      {
        return true;
      }
      else if (ivar == IY and
               is_edge_at_block_border<dim>(edge_indexes, m_block_sizes, Face::ZMAX) and
               conformal_full_status_t<dim>::get_status(Face::ZMAX, status) ==
                 conformal_neighbor_status::NEIGHBOR_IS_FINER)
      {
        return true;
      }

      // Z edges
      if (ivar == IZ and
          is_edge_at_block_edge<dim>(edge_indexes, m_block_sizes, Face::XMIN, Face::YMIN) and
          conformal_full_status_t<dim>::get_status(Face::XMIN, Face::YMIN, status) ==
            conformal_neighbor_status::NEIGHBOR_IS_FINER)
      {
        return true;
      }
      else if (ivar == IZ and
               is_edge_at_block_edge<dim>(edge_indexes, m_block_sizes, Face::XMAX, Face::YMIN) and
               conformal_full_status_t<dim>::get_status(Face::XMAX, Face::YMIN, status) ==
                 conformal_neighbor_status::NEIGHBOR_IS_FINER)
      {
        return true;
      }
      else if (ivar == IZ and
               is_edge_at_block_edge<dim>(edge_indexes, m_block_sizes, Face::XMIN, Face::YMAX) and
               conformal_full_status_t<dim>::get_status(Face::XMIN, Face::YMAX, status) ==
                 conformal_neighbor_status::NEIGHBOR_IS_FINER)
      {
        return true;
      }
      else if (ivar == IZ and
               is_edge_at_block_edge<dim>(edge_indexes, m_block_sizes, Face::XMAX, Face::YMAX) and
               conformal_full_status_t<dim>::get_status(Face::XMAX, Face::YMAX, status) ==
                 conformal_neighbor_status::NEIGHBOR_IS_FINER)
      {
        return true;
      }
      else if (ivar == IZ and
               is_edge_at_block_border<dim>(edge_indexes, m_block_sizes, Face::XMIN) and
               conformal_full_status_t<dim>::get_status(Face::XMIN, status) ==
                 conformal_neighbor_status::NEIGHBOR_IS_FINER)
      {
        return true;
      }
      else if (ivar == IZ and
               is_edge_at_block_border<dim>(edge_indexes, m_block_sizes, Face::XMAX) and
               conformal_full_status_t<dim>::get_status(Face::XMAX, status) ==
                 conformal_neighbor_status::NEIGHBOR_IS_FINER)
      {
        return true;
      }
      else if (ivar == IZ and
               is_edge_at_block_border<dim>(edge_indexes, m_block_sizes, Face::YMIN) and
               conformal_full_status_t<dim>::get_status(Face::YMIN, status) ==
                 conformal_neighbor_status::NEIGHBOR_IS_FINER)
      {
        return true;
      }
      else if (ivar == IZ and
               is_edge_at_block_border<dim>(edge_indexes, m_block_sizes, Face::YMAX) and
               conformal_full_status_t<dim>::get_status(Face::YMAX, status) ==
                 conformal_neighbor_status::NEIGHBOR_IS_FINER)
      {
        return true;
      }

      return false;
    }
  } // is_neighbor_finer

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION void
  operator()(const index_t & global_index) const
  {

    // convert global index into
    // - octant id
    // - edge_flat_index inside block
    //
    const auto & nbEdgesPerLeaf = m_vector_potential.num_elements_per_octant();

    const auto iOct = global_index / nbEdgesPerLeaf;
    const auto edge_flat_index = static_cast<int32_t>(global_index - iOct * nbEdgesPerLeaf);

    // compute ix,iy,iz,ivar of local face inside
    // block from a face flat-index
    const auto edge_indexes =
      edge_flat_index_unravel<dim>(edge_flat_index, m_block_sizes, m_vector_potential.offsets());

    auto const & bSize = m_block_sizes[IX];

    // get block orchard key
    const auto key = m_orchard_keys(iOct);

    // get block level
    const auto level = orchard_key_t<dim>::level(key);

    // compute physical x,y,z for that face center
    const auto xyz = orchard_key_to_edgecenter_real_space<dim>(
      key, edge_indexes, bSize, m_scaling_factor, m_xyz_min);

    const auto status = m_conformal_full_status_view(iOct);

    // same size in all direction
    const auto dx = compute_cell_length<dim>(level, m_block_sizes[IX]) * m_scaling_factor;
    // const auto & dy = dx;
    // const auto & dz = dx;

    // remember that in 2D only Ax and Ay may need to be modified, in case neighbor is at finer
    // resolution
    if (is_neighbor_finer(edge_indexes, status))
    {
      auto const & ivar = edge_indexes[dim];

      auto xyz1 = xyz;
      xyz1[ivar] = xyz[ivar] - KALYPSSO_NUM(0.25) * dx;
      auto xyz2 = xyz;
      xyz2[ivar] = xyz[ivar] + KALYPSSO_NUM(0.25) * dx;

      m_vector_potential(edge_indexes, iOct) =
        KALYPSSO_NUM(0.5) *
        (m_ana_vector_potential(xyz1, ivar) + m_ana_vector_potential(xyz2, ivar));
    }

  } // operator()

}; // class FixVectorPotentialAtNonConformalEdges

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_MHD_CT_INIT_FIX_VECTOR_POTENTIAL_AT_NON_CONFORMAL_EDGES_H_
