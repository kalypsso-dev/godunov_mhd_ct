// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitOrszagTang.cpp
 */

#include <godunov_mhd_ct/init/InitOrszagTang.h>
#include <godunov_mhd_ct/SolverGodunovMHD.h>

#include <kalypsso/core/orchard_key_utils.h>
#include <kalypsso/core/problems/init_cond_utils.h>

namespace kalypsso
{

namespace godunov_mhd_ct
{

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
InitOrszagTangDataFunctor<dim, device_t>::InitOrszagTangDataFunctor(
  DataArrayBlock_t             Udata,
  FaceDataArrayBlock_t         Bface,
  FieldMap<models::MHD>  fm,
  orchard_key_view_t<device_t> orchard_keys,
  int32_t                      local_num_octants,
  ConfigMap const &            config_map)
  : m_Udata(Udata)
  , m_Bface(Bface)
  , m_fm(fm)
  , m_orchard_keys(orchard_keys)
  , m_local_num_octants(local_num_octants)
  , m_mhd_settings(config_map)
  , m_otParams(config_map)
  , m_scaling_factor(get_scaling_factor(config_map))
  , m_xyz_min(get_xyz_min<dim>(config_map))
  , m_xyz_max(get_xyz_max<dim>(config_map)){};

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitOrszagTangDataFunctor<dim, device_t>::apply(DataArrayBlock_t             Udata,
                                                FaceDataArrayBlock_t         Bface,
                                                FieldMap<models::MHD>  fm,
                                                orchard_key_view_t<device_t> orchard_keys,
                                                int32_t                      local_num_octants,
                                                ConfigMap const &            config_map)
{
  // data init functor
  InitOrszagTangDataFunctor functor(Udata, Bface, fm, orchard_keys, local_num_octants, config_map);

  // for cell center init
  const auto nbCellsPerLeaf = Udata.num_cells();
  const auto nbCellsTotal = local_num_octants * nbCellsPerLeaf;

  // for face center init
  const auto nbFacesPerLeaf = Bface.num_elements_per_octant();
  const auto nbFacesTotal = local_num_octants * nbFacesPerLeaf;

  // initialize hydro variables but total energy (cell-centered)
  Kokkos::parallel_for(
    "kalypsso::godunov_mhd_ct::InitOrszagTangDataFunctor - all hydro variables except total energy",
    Kokkos::RangePolicy<exec_space, TagInitHydroVar>(0, nbCellsTotal),
    functor);

  // initialize mag field (face-centered)
  Kokkos::parallel_for("kalypsso::godunov_mhd_ct::InitOrszagTangDataFunctor - magnetic field",
                       Kokkos::RangePolicy<exec_space, TagInitMagField>(0, nbFacesTotal),
                       functor);

  // initialize total energy (cell-centered)
  Kokkos::parallel_for("kalypsso::godunov_mhd_ct::InitOrszagTangDataFunctor - total energy",
                       Kokkos::RangePolicy<exec_space, TagInitTotalEnergy>(0, nbCellsTotal),
                       functor);

} // apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitOrszagTangDataFunctor<dim, device_t>::operator()(TagInitHydroVar,
                                                     const int32_t & global_index) const
{

  // convert global index into
  // - octant id
  // - cell_index inside block (from 0 to nbCellsPerLeaf-1)
  const auto iOct = global_index / m_Udata.num_cells();
  const auto cell_index = global_index - iOct * m_Udata.num_cells();

  constexpr auto ID = models::MHD::ID;
  constexpr auto IU = models::MHD::IU;
  constexpr auto IV = models::MHD::IV;
  constexpr auto IW = models::MHD::IW;

  // Orszag-Tang vortex problem parameters
  const real_t     gamma0 = m_mhd_settings.hydro.gamma0;
  constexpr real_t TWO_PI_F = 2 * PI_F;
  const real_t     p0 = gamma0 / (TWO_F * TWO_PI_F);
  const real_t     d0 = gamma0 * p0;
  constexpr real_t v0 = ONE_F;
  auto const &     vortex_dir = m_otParams.vortex_dir;

  const auto & block_sizes = m_Udata.block_size();

  // compute ix,iy,iz of local cell inside
  // block from index
  const auto iCoord = cellindex_to_coord<dim>(cell_index, block_sizes);

  // get block orchard key
  const auto key = m_orchard_keys(iOct);

  auto const & bSize = block_sizes[IX];

  // compute physical x,y,z at cell center
  const auto xyz =
    orchard_key_to_cellcenter_real_space<dim>(key, iCoord, bSize, m_scaling_factor, m_xyz_min);

  if constexpr (dim == 2)
  {
    // clang-format off
    m_Udata(cell_index, m_fm[ID], iOct) =  d0;
    m_Udata(cell_index, m_fm[IU], iOct) = -d0 * v0 * sin(xyz[IY] * TWO_PI_F);
    m_Udata(cell_index, m_fm[IV], iOct) =  d0 * v0 * sin(xyz[IX] * TWO_PI_F);
    m_Udata(cell_index, m_fm[IW], iOct) = ZERO_F;
    // clang-format on
  }
  else if (dim == 3)
  {
    if (vortex_dir == OrszagTangParams::VortexDir::Z)
    {
      // clang-format off
      m_Udata(cell_index, m_fm[ID], iOct) =  d0;
      m_Udata(cell_index, m_fm[IU], iOct) = -d0 * v0 * sin(xyz[IY] * TWO_PI_F);
      m_Udata(cell_index, m_fm[IV], iOct) =  d0 * v0 * sin(xyz[IX] * TWO_PI_F);
      m_Udata(cell_index, m_fm[IW], iOct) = ZERO_F;
      // clang-format on
    }
    else if (vortex_dir == OrszagTangParams::VortexDir::X)
    {
      // clang-format off
      m_Udata(cell_index, m_fm[ID], iOct) =  d0;
      m_Udata(cell_index, m_fm[IV], iOct) = -d0 * v0 * sin(xyz[IZ] * TWO_PI_F);
      m_Udata(cell_index, m_fm[IW], iOct) =  d0 * v0 * sin(xyz[IY] * TWO_PI_F);
      m_Udata(cell_index, m_fm[IU], iOct) = ZERO_F;
      // clang-format on
    }
    else if (vortex_dir == OrszagTangParams::VortexDir::Y)
    {
      // clang-format off
      m_Udata(cell_index, m_fm[ID], iOct) =  d0;
      m_Udata(cell_index, m_fm[IW], iOct) = -d0 * v0 * sin(xyz[IX] * TWO_PI_F);
      m_Udata(cell_index, m_fm[IU], iOct) =  d0 * v0 * sin(xyz[IZ] * TWO_PI_F);
      m_Udata(cell_index, m_fm[IV], iOct) = ZERO_F;
      // clang-format on
    }
  }

} // end operator() - TagInitHydroVar

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitOrszagTangDataFunctor<dim, device_t>::operator()(TagInitMagField,
                                                     const int32_t & global_index) const
{

  // convert global index into
  // - octant id
  // - face_flat_index inside block (from 0 to m_nbFacesPerLeaf-1)
  //
  // please remember that is slightly too large, so we need to protect write access to only valid
  // multi-index i,j,k
  const auto iOct = global_index / m_Bface.num_elements_per_octant();
  const auto face_flat_index =
    static_cast<int32_t>(global_index - iOct * m_Bface.num_elements_per_octant());

  const auto & block_sizes = m_Udata.block_size();

  // compute ix,iy,iz,ivar of local face inside
  // block from a face flat-index
  const auto face_indexes =
    face_flat_index_unravel<dim>(face_flat_index, block_sizes, m_Bface.offsets(), m_Bface.shift());

  // Orszag-Tang vortex problem parameters
  constexpr real_t TWO_PI_F = 2 * PI_F;
  const real_t     B0 = ONE_F / sqrt(TWO_F * TWO_PI_F);
  auto const &     kt = m_otParams.kt;
  auto const &     vortex_dir = m_otParams.vortex_dir;
  const auto &     xmin = m_xyz_min[IX];
  const auto &     xmax = m_xyz_max[IX];
  const auto &     ymin = m_xyz_min[IY];
  const auto &     ymax = m_xyz_max[IY];

  auto const & bSize = block_sizes[IX];

  // get block orchard key
  const auto key = m_orchard_keys(iOct);

  // compute physical x,y,z for that face center
  const auto xyz = orchard_key_to_facecenter_real_space<dim>(
    key, face_indexes, bSize, m_scaling_factor, m_xyz_min);

  if constexpr (dim == 2)
  {

    auto const & i = face_indexes[IX];
    auto const & j = face_indexes[IY];
    auto const & ivar = face_indexes[dim];

    if (ivar == IX)
    {
      // Bx on X-face
      m_Bface(i, j, IX, iOct) = -B0 * sin(xyz[IY] * TWO_PI_F);
    }
    else if (ivar == IY)
    {
      // By on Y-face
      m_Bface(i, j, IY, iOct) = B0 * sin(2 * xyz[IX] * TWO_PI_F);
    }
    else if (ivar == IZ)
    {
      m_Bface(i, j, IZ, iOct) = ZERO_F;
    }
  }
  else if constexpr (dim == 3)
  {
    auto const & i = face_indexes[IX];
    auto const & j = face_indexes[IY];
    auto const & k = face_indexes[IZ];
    auto const & ivar = face_indexes[dim];

    const auto & zmin = m_xyz_min[IZ];
    const auto & zmax = m_xyz_max[IZ];

    if (vortex_dir == OrszagTangParams::VortexDir::Z)
    {
      if (ivar == IX)
      {
        // Bx on X-face

        // clang-format off
        m_Bface(i, j, k, IX, iOct) =
          -B0 * cos(2 * TWO_PI_F * kt * (xyz[IZ] - zmin) / (zmax - zmin)) * sin(xyz[IY] * TWO_PI_F);
        // clang-format on
      }
      else if (ivar == IY)
      {
        // By on Y-face

        // clang-format off
        m_Bface(i, j, k, IY, iOct) =
          B0 * cos(2 * TWO_PI_F * kt * (xyz[IZ] - zmin) / (zmax - zmin)) * sin(2 * xyz[IX] * TWO_PI_F);
        // clang-format on
      }
      else if (ivar == IZ)
      {
        // Bz on Z-face
        m_Bface(i, j, k, IZ, iOct) = ZERO_F;
      }
    }
    else if (vortex_dir == OrszagTangParams::VortexDir::X)
    {
      if (ivar == IY)
      {
        // By on Y-face

        // clang-format off
        m_Bface(i, j, k, IY, iOct) =
          -B0 * cos(2 * TWO_PI_F * kt * (xyz[IX] - xmin) / (xmax - xmin)) * sin(xyz[IZ] * TWO_PI_F);
        // clang-format on
      }
      else if (ivar == IZ)
      {
        // Bz on Z-face

        // clang-format off
        m_Bface(i, j, k, IZ, iOct) =
          B0 * cos(2 * TWO_PI_F * kt * (xyz[IX] - xmin) / (xmax - xmin)) * sin(2 * xyz[IY] * TWO_PI_F);
        // clang-format on
      }
      else if (ivar == IX)
      {
        // Bx on X-face
        m_Bface(i, j, k, IX, iOct) = ZERO_F;
      }
    }
    else if (vortex_dir == OrszagTangParams::VortexDir::Y)
    {
      if (ivar == IZ)
      {
        // Bz on Z-face

        // clang-format off
        m_Bface(i, j, k, IZ, iOct) =
          -B0 * cos(2 * TWO_PI_F * kt * (xyz[IY] - ymin) / (ymax - ymin)) * sin(xyz[IX] * TWO_PI_F);
        // clang-format on
      }
      else if (ivar == IX)
      {
        // Bx on X-face

        // clang-format off
        m_Bface(i, j, k, IX, iOct) =
          B0 * cos(2 * TWO_PI_F * kt * (xyz[IY] - ymin) / (ymax - ymin)) * sin(2 * xyz[IZ] * TWO_PI_F);
        // clang-format on
      }
      else if (ivar == IY)
      {
        // By on Y-face
        m_Bface(i, j, k, IY, iOct) = ZERO_F;
      }
    }
  } // end 3d

} // end operator() - TagInitMagField

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitOrszagTangDataFunctor<dim, device_t>::operator()(TagInitTotalEnergy,
                                                     const int32_t & global_index) const
{
  constexpr auto ID = models::MHD::ID;
  constexpr auto IU = models::MHD::IU;
  constexpr auto IV = models::MHD::IV;
  constexpr auto IW = models::MHD::IW;
  constexpr auto IE = models::MHD::IE;

  // convert global index into
  // - octant id
  // - cell_index inside block (from 0 to nbCellsPerLeaf-1)
  const auto iOct = global_index / m_Udata.num_cells();
  const auto cell_index = global_index - iOct * m_Udata.num_cells();

  // Orszag-Tang vortex problem parameters
  constexpr real_t TWO_PI_F = 2 * PI_F;
  const real_t     gamma0 = m_mhd_settings.hydro.gamma0;
  const real_t     p0 = gamma0 / (TWO_F * TWO_PI_F);

  const auto & block_sizes = m_Udata.block_size();

  // compute ix,iy,iz of local cell inside
  // block from index
  auto iCoord = cellindex_to_coord<dim>(cell_index, block_sizes);

  // total energy
  real_t total_energy = p0 / (gamma0 - ONE_F);

  // add kinetic energy
  if constexpr (dim == 2)
  {
    total_energy += HALF_F *
                    (m_Udata(cell_index, m_fm[IU], iOct) * m_Udata(cell_index, m_fm[IU], iOct) +
                     m_Udata(cell_index, m_fm[IV], iOct) * m_Udata(cell_index, m_fm[IV], iOct)) /
                    m_Udata(cell_index, m_fm[ID], iOct);
  }
  else
  {
    total_energy += HALF_F *
                    (m_Udata(cell_index, m_fm[IU], iOct) * m_Udata(cell_index, m_fm[IU], iOct) +
                     m_Udata(cell_index, m_fm[IV], iOct) * m_Udata(cell_index, m_fm[IV], iOct) +
                     m_Udata(cell_index, m_fm[IW], iOct) * m_Udata(cell_index, m_fm[IW], iOct)) /
                    m_Udata(cell_index, m_fm[ID], iOct);
  }

  // add magnetic energy
  if constexpr (dim == 2)
  {
    auto const & i = iCoord[IX];
    auto const & j = iCoord[IY];

    const real_t a = HALF_F * (m_Bface(i, j, IX, iOct) + m_Bface(i + 1, j, IX, iOct));
    const real_t b = HALF_F * (m_Bface(i, j, IY, iOct) + m_Bface(i, j + 1, IY, iOct));
    const real_t c = m_Bface(i, j, IZ, iOct);
    total_energy += HALF_F * (a * a + b * b + c * c);
  }
  else
  {
    auto const & i = iCoord[IX];
    auto const & j = iCoord[IY];
    auto const & k = iCoord[IZ];

    const real_t a = HALF_F * (m_Bface(i, j, k, IX, iOct) + m_Bface(i + 1, j, k, IX, iOct));
    const real_t b = HALF_F * (m_Bface(i, j, k, IY, iOct) + m_Bface(i, j + 1, k, IY, iOct));
    const real_t c = HALF_F * (m_Bface(i, j, k, IZ, iOct) + m_Bface(i, j, k + 1, IZ, iOct));
    total_energy += HALF_F * (a * a + b * b + c * c);
  }

  m_Udata(cell_index, m_fm[IE], iOct) = total_energy;

} // end operator() - TagInitTotalEnergy

// explicit template instantiation
template class InitOrszagTangDataFunctor<2, kalypsso::DefaultDevice>;
template class InitOrszagTangDataFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitOrszagTang<dim, device_t>::apply(SolverGodunovMHD<dim, device_t> & solver)
{

  auto              amr_mesh = solver.amr_mesh();
  ConfigMap const & config_map = solver.config_map();
  const int         level_min = solver.hydro_params().level_min;
  const int         level_max = solver.hydro_params().level_max;

  // OrszagTangParams otParams = OrszagTangParams(config_map);
  //  field manager index array
  //  auto fm = solver.model().get_fieldmap();

  constexpr bool do_reset_ghosts = true;
  solver.update_mesh(do_reset_ghosts);

  // resize Udata
  solver.resize_solver_data();

  // first init of Udata
  InitOrszagTangDataFunctor<dim, device_t>::apply(solver.U(),
                                                  solver.Bface(),
                                                  solver.model().get_fieldmap(),
                                                  solver.mesh_map()->orchard_keys(),
                                                  solver.amr_mesh()->local_num_quadrants(),
                                                  config_map);

  // iterate several refinements
  int level = level_min;
  while (level < level_max)
  {
    //
    // 1. apply amr cycle using regular refine criterion
    //
    solver.do_amr_cycle();

    //
    // 2. update Udata
    //
    InitOrszagTangDataFunctor<dim, device_t>::apply(solver.U(),
                                                    solver.Bface(),
                                                    solver.model().get_fieldmap(),
                                                    solver.mesh_map()->orchard_keys(),
                                                    solver.amr_mesh()->local_num_quadrants(),
                                                    config_map);


    // update level
    ++level;

  } // end while level<level_max

#ifdef KALYPSSO_CORE_USE_MPI
  // load balancing (= repartitioning) the octree mesh + userdata over the MPI processes.
  // U and U2 will be resized
  solver.do_load_balancing();
#endif

} // InitOrszagTang::apply

// explicit template instantiation declaration to prevent implicit instantiation
template class InitOrszagTang<2, kalypsso::DefaultDevice>;
template class InitOrszagTang<3, kalypsso::DefaultDevice>;

} // namespace godunov_mhd_ct

} // namespace kalypsso
