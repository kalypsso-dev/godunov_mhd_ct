// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file FillOutside.cpp
 *
 * Implement border conditions (other than periodic) for MHD.
 */
#include <godunov_mhd_ct/border_conditions/FillOutside.h>
#include <kalypsso/core/mesh_utils.h> // for definition of Face::XMIN, etc...

#include <kalypsso/core/MeshMap.h> //for key_to_value

namespace kalypsso
{

namespace godunov_mhd_ct
{


// ==============================================================================
// ==============================================================================
template <size_t dim, typename device_t>
FillOutsideCellFunctor<dim, device_t>::FillOutsideCellFunctor(
  DataArrayBlock_t const &     hydro,
  FaceDataArrayBlock_t const & Bface,
  AMRMeshInfo const &          amr_mesh_info,
  orchard_key_view_t const &   orchard_keys,
  amr_hashmap_t const &        amr_hashmap,
  FieldMap<MHD> const &        fm,
  ConfigMap const &            config_map,
  ParallelEnv const &          par_env)
  : m_hydro(hydro)
  , m_Bface(Bface)
  , m_amr_mesh_info(amr_mesh_info)
  , m_orchard_keys_device(orchard_keys)
  , m_amr_hashmap_device(amr_hashmap)
  , m_fm(fm)
  , m_brick_size(get_brick_sizes<dim>(config_map))
  , m_brick_periodicity(get_brick_periodicity<dim>(config_map))
  , m_bc_types(BorderConditionsConfig<BC_MHD>::read_border_condition<dim>(config_map, par_env))
{}

// ==============================================================
// ==============================================================
//! static method which does it all: create and execute functor with range policy
template <size_t dim, typename device_t>
void
FillOutsideCellFunctor<dim, device_t>::apply(DataArrayBlock_t const &     hydro,
                                             FaceDataArrayBlock_t const & Bface,
                                             AMRMeshInfo const &          amr_mesh_info,
                                             orchard_key_view_t const &   orchard_keys,
                                             amr_hashmap_t const &        amr_hashmap,
                                             FieldMap<MHD> const &        fm,
                                             ConfigMap const &            config_map,
                                             ParallelEnv const &          par_env)
{

  // create compute functor
  FillOutsideCellFunctor<dim, device_t> functor(
    hydro, Bface, amr_mesh_info, orchard_keys, amr_hashmap, fm, config_map, par_env);

  const int32_t start_octant = amr_mesh_info.first_outside_quad_local_id();
  const int32_t end_octant = start_octant + amr_mesh_info.total_local_number_of_outside_quads();

  //
  // fill cell-centered variables (hydro)
  //

  {
    const int32_t nb_cells_per_leaf = hydro.num_cells();
    const int32_t start = start_octant * nb_cells_per_leaf;
    const int32_t end = end_octant * nb_cells_per_leaf;

    Kokkos::parallel_for("FillOutsideFunctor - hydro variables (cell-centered)",
                         Kokkos::RangePolicy<exec_space, TagFillHydroVar>(start, end),
                         functor);
  }

  //
  // fill face-center variables (magnetic field)
  //
  {
    const auto    nb_faces_per_leaf = Bface.num_elements_per_octant();
    const int32_t start = start_octant * nb_faces_per_leaf;
    const int32_t end = end_octant * nb_faces_per_leaf;

    Kokkos::parallel_for("FillOutsideFunctor - magnetic field (face-centered)",
                         Kokkos::RangePolicy<exec_space, TagFillMagField>(start, end),
                         functor);
  }

} // apply

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
FillOutsideCellFunctor<dim, device_t>::operator()(TagFillHydroVar const &,
                                                  const index_t & i_global) const
{

  const auto block_size = m_hydro.block_size();
  const auto nb_cells_per_leaf = m_hydro.num_cells();

  // iOct_local, by design, is associated to an outside quadrant
  const auto i_oct_outside = i_global / nb_cells_per_leaf;

  const auto i_cell_outside = i_global - nb_cells_per_leaf * i_oct_outside;
  const auto coord_out = cellindex_to_coord<dim>(i_cell_outside, block_size);

  // get orchard key corresponding to visited outside quadrants
  const auto key_outside = m_orchard_keys_device(i_oct_outside);

  // when computing inside quadrant key, we always use a "virtual" key computed as the
  // periodic image of the outside quadrant; so here we need is_periodic to be array of
  // "true"
  constexpr auto is_periodic = get_bool_array<dim>(true);

  // quadrant is at domain border, compute outside normal at the periodic image of key_outside
  auto outside_normal =
    orchard_key_t<dim>::get_outside_normal(key_outside, m_brick_size, is_periodic);

  // the following code just ensure that outside normal is non-zero vector only for quadrant that
  // are actually outside domain
  outside_normal[IX] *= orchard_key_t<dim>::is_touching_face_X(key_outside);
  outside_normal[IY] *= orchard_key_t<dim>::is_touching_face_Y(key_outside);
  if constexpr (dim == 3)
  {
    outside_normal[IZ] *= orchard_key_t<dim>::is_touching_face_Z(key_outside);
  }

  // compute orchard key of corresponding inside quadrant (just across external border, i.e. along
  // outside normal, but opposite direction)
  auto key_inside = orchard_key_t<dim>::get_neighbor_key_same_level(
    key_outside, outside_normal, m_brick_size, m_brick_periodicity);
  orchard_key_t<dim>::reset_outside_bits(key_inside);

  // get iOct_inside from the unordered map
  const auto key_status = key_to_value(key_inside, m_amr_hashmap_device);

  // check AMR key validity
  {
    [[maybe_unused]] auto const & is_valid_key = key_status.first;

    // make sure key_inside actually exist in map
    // if it doesn't we have a serious logical problem
    KOKKOS_ASSERT(is_valid_key &&
                  "[kalypsso::FillOutsideCellFunctor] key_inside does not exist in hashmap !?");
  }

  const auto i_oct_inside = key_status.second;

  // make sure i_oct_inside is actually inside
  KOKKOS_ASSERT(
    i_oct_inside < m_amr_mesh_info.first_outside_quad_local_id() &&
    "[kalypsso::FillOutsideCellFunctor] i_oct_inside does not identify an octant inside domain ?!");

  //
  // now we can fill outside hydro variables according to border condition
  //

  // get list of faces
  const auto faces = Face::get_all_faces<dim>();

  for (const auto face : faces)
  {
    // normal direction
    const Dir::dir_t dir = face / 2;

    if (orchard_key_t<dim>::is_at_domain_border(key_inside, face, m_brick_size) and
        orchard_key_t<dim>::is_outside_dir(key_outside, dir))
    {
      if (m_bc_types[face]._to_integral() == +BC_MHD::ZERO_GRADIENT)
      {
        auto coord_in = coord_out;
        coord_in[dir] = Face::is_left_face(face) ? 0 : block_size[dir] - 1;
        auto i_cell_inside = coord_to_cellindex<dim>(coord_in, block_size);

        for (int32_t ivar = 0; ivar < m_hydro.num_vars(); ++ivar)
        {
          m_hydro(i_cell_outside, ivar, i_oct_outside) = m_hydro(i_cell_inside, ivar, i_oct_inside);
        }
      }
      else if (m_bc_types[face]._to_integral() == +BC_MHD::WALL)
      {
        auto coord_in = coord_out;
        coord_in[dir] = block_size[dir] - 1 - coord_out[dir];
        auto i_cell_inside = coord_to_cellindex<dim>(coord_in, block_size);

        // copy data and negate normal velocity
        for (int32_t ivar = 0; ivar < m_hydro.num_vars(); ++ivar)
        {
          m_hydro(i_cell_outside, ivar, i_oct_outside) = m_hydro(i_cell_inside, ivar, i_oct_inside);
        }

        m_hydro(i_cell_outside, m_fm[MHD::VarId(MHD::IU + dir)], i_oct_outside) *= -1;

      } // end BC_MHD::WALL
    } // end if is_at_domain_border
  } // for faces

} // operator() - TagFillHydroVar

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
FillOutsideCellFunctor<dim, device_t>::operator()(TagFillMagField const &,
                                                  const index_t & i_global) const
{
  const auto block_size = m_hydro.block_size();
  const auto nb_faces_per_leaf = m_Bface.num_elements_per_octant();

  // i_oct_local, by design, is associated to an outside quadrant
  const auto i_oct_outside =
    i_global / nb_faces_per_leaf; // + m_amr_mesh_info.first_outside_quad_local_id();

  const auto face_flatindex_out =
    static_cast<int32_t>(i_global - nb_faces_per_leaf * i_oct_outside);

  // compute ix,iy,iz,ivar of local face inside
  // block from a face flat-index
  const auto face_indexes_out = face_flat_index_unravel<dim>(
    face_flatindex_out, block_size, m_Bface.offsets(), m_Bface.shift());

  const auto ivar = face_indexes_out[dim];

  // get orchard key corresponding to visited outside quadrants
  const auto key_outside = m_orchard_keys_device(i_oct_outside);

  // when computing inside quadrant key, we always use a "virtual" key computed as the
  // periodic image of the outside quadrant; so here we need is_periodic to be array of
  // "true"
  constexpr auto is_periodic = get_bool_array<dim>(true);

  // quadrant is at domain border, compute outside normal at the periodic image of key_outside
  auto outside_normal =
    orchard_key_t<dim>::get_outside_normal(key_outside, m_brick_size, is_periodic);

  // the following code just ensure that outside normal is non-zero vector only for quadrant that
  // are actually outside domain
  outside_normal[IX] *= orchard_key_t<dim>::is_touching_face_X(key_outside);
  outside_normal[IY] *= orchard_key_t<dim>::is_touching_face_Y(key_outside);
  if constexpr (dim == 3)
  {
    outside_normal[IZ] *= orchard_key_t<dim>::is_touching_face_Z(key_outside);
  }

  // compute orchard key of corresponding inside quadrant (just across external border, i.e. along
  // outside normal, but opposite direction)
  auto key_inside = orchard_key_t<dim>::get_neighbor_key_same_level(
    key_outside, outside_normal, m_brick_size, m_brick_periodicity);
  orchard_key_t<dim>::reset_outside_bits(key_inside);

  // get i_oct_inside from the unordered map
  const auto key_status = key_to_value(key_inside, m_amr_hashmap_device);

  // check AMR key validity
  {
    [[maybe_unused]] auto const & is_valid_key = key_status.first;

    // make sure key_inside actually exist in map
    // if it doesn't we have a serious logical problem
    KOKKOS_ASSERT(is_valid_key &&
                  "[kalypsso::FillOutsideCellFunctor] key_inside does not exist in hashmap !?");
  }

  const auto i_oct_inside = key_status.second;

  // cross-check i_oct_inside is actually inside
  KOKKOS_ASSERT(
    i_oct_inside < m_amr_mesh_info.first_outside_quad_local_id() &&
    "[kalypsso::FillOutsideCellFunctor] i_oct does not identify an octant inside domain ?!");

  //
  // now we can fill outside magnetic field variables according to border condition
  //

  // outside_normal is a vector that points inside the domain

  auto face_indexes_in = face_indexes_out;

  // deal with faces XMIN/XMAX
  if (outside_normal[IX] > 0)
  {
    // touching XMIN
    if (m_bc_types[XMIN]._to_integral() == +BC_MHD::ZERO_GRADIENT)
    {
      face_indexes_in[IX] = 0;
      // face_indexes_in[IX] = (m_Bface.face_block_size(ivar)[IX] - 1 - face_indexes_out[IX]);
    }
    else if (m_bc_types[XMIN]._to_integral() == +BC_MHD::WALL)
    {
      // TODO
    }
  }
  else if (outside_normal[IX] < 0)
  {
    // touching XMAX
    if (m_bc_types[XMAX]._to_integral() == +BC_MHD::ZERO_GRADIENT)
    {
      face_indexes_in[IX] = m_Bface.face_block_size(ivar)[IX] - 1;
      // face_indexes_in[IX] = (m_Bface.face_block_size(ivar)[IX] - 1 - face_indexes_out[IX]);
    }
    else if (m_bc_types[XMAX]._to_integral() == +BC_MHD::WALL)
    {
      // TODO
    }
  }

  // deal with faces YMIN/YMAX
  if (outside_normal[IY] > 0)
  {
    // touching YMIN
    if (m_bc_types[YMIN]._to_integral() == +BC_MHD::ZERO_GRADIENT)
    {
      face_indexes_in[IY] = 0;
      // face_indexes_in[IY] = (m_Bface.face_block_size(ivar)[IY] - 1 - face_indexes_out[IY]);
    }
    else if (m_bc_types[YMIN]._to_integral() == +BC_MHD::WALL)
    {
      // TODO
    }
  }
  else if (outside_normal[IY] < 0)
  {
    // touching YMAX
    if (m_bc_types[YMAX]._to_integral() == +BC_MHD::ZERO_GRADIENT)
    {
      face_indexes_in[IY] = m_Bface.face_block_size(ivar)[IY] - 1;
      // face_indexes_in[IY] = (m_Bface.face_block_size(ivar)[IY] - 1 - face_indexes_out[IY]);
    }
    else if (m_bc_types[YMAX]._to_integral() == +BC_MHD::WALL)
    {
      // TODO
    }
  }

  if constexpr (dim == 3)
  {
    // deal with faces ZMIN/ZMAX
    if (outside_normal[IZ] > 0)
    {
      // touching ZMIN
      if (m_bc_types[ZMIN]._to_integral() == +BC_MHD::ZERO_GRADIENT)
      {
        face_indexes_in[IZ] = 0;
        // face_indexes_in[IZ] = m_Bface.face_block_size(ivar)[IZ] - 1 - (face_indexes_out[IZ] % 2);
      }
      else if (m_bc_types[ZMIN]._to_integral() == +BC_MHD::WALL)
      {
        // TODO
      }
    }
    else if (outside_normal[IZ] < 0)
    {
      // touching ZMAX
      if (m_bc_types[ZMAX]._to_integral() == +BC_MHD::ZERO_GRADIENT)
      {
        face_indexes_in[IZ] = m_Bface.face_block_size(ivar)[IZ] - 1;
        // face_indexes_in[IZ] = m_Bface.face_block_size(ivar)[IZ] - 1 - (face_indexes_out[IZ] % 2);
      }
      else if (m_bc_types[ZMAX]._to_integral() == +BC_MHD::WALL)
      {
        // TODO
      }
    }
  }

  // printf("KKK out i=%d j=%d var=%d ioct=%d | normal %d %d | in ioct=%d i=%d j=%d var=%d\n",
  //        face_indexes_out[IX],
  //        face_indexes_out[IY],
  //        face_indexes_out[dim],
  //        i_oct_outside,
  //        outside_normal[IX],
  //        outside_normal[IY],
  //        i_oct_inside,
  //        face_indexes_in[IX],
  //        face_indexes_in[IY],
  //        face_indexes_in[dim]);

  // copy value
  if constexpr (dim == 2)
  {
    m_Bface(face_indexes_out[IX], face_indexes_out[IY], ivar, i_oct_outside) =
      m_Bface(face_indexes_in[IX], face_indexes_in[IY], ivar, i_oct_inside);
  }
  else if constexpr (dim == 3)
  {
    m_Bface(face_indexes_out[IX], face_indexes_out[IY], face_indexes_out[IZ], ivar, i_oct_outside) =
      m_Bface(face_indexes_in[IX], face_indexes_in[IY], face_indexes_in[IZ], ivar, i_oct_inside);
  }

} // operator() - TagFillMagField

// explicit template instantiation
template class FillOutsideCellFunctor<2, kalypsso::DefaultDevice>;
template class FillOutsideCellFunctor<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso
