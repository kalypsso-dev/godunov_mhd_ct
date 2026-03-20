// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file MHDProfilingData.h
 */
#ifndef KALYPSSO_GODUNOV_MHD_PROFILING_DATA_H_
#define KALYPSSO_GODUNOV_MHD_PROFILING_DATA_H_

#include <kalypsso/utils/monitoring/ProfilingManager.h>
#include <../better-enums/enum.h>

#define ADD_CASE(enum, name, color) \
  case ProfilingZone::enum:         \
    return                          \
    {                               \
      name, Color_t::color()        \
    }

#ifndef KALYPSSO_PROFILING_REGION
#  define KALYPSSO_PROFILING_REGION(mgr, value) \
    ProfilingTimer region_##value(mgr, ProfilingZone::value)
#endif

namespace kalypsso
{
namespace godunov_mhd_ct
{

// clang-format off
BETTER_ENUM(ProfilingZone, uint32_t,
  AMR_CYCLE,
    AMR_CYCLE_RESIZE_AUX,
    AMR_CYCLE_COPY_HASHMAP,
    AMR_CYCLE_SYNC_MPI_GHOSTS,
    AMR_CYCLE_MARK_CELLS,
    AMR_CYCLE_ADAPT_MESH,
    AMR_CYCLE_USERDATA_REMAP,
  LOAD_BALANCING,
    LOAD_BALANCING_PARTITION_MESH,
    LOAD_BALANCING_PARTITION_USERDATA,
    LOAD_BALANCING_UPDATE_MESH,
    LOAD_BALANCING_RESIZE,
  NUM,
    NUM_CFL,
    NUM_SCHEME,
      NUM_SCHEME_CONV_PRIM,
      NUM_SCHEME_Q_GHOSTED,
      NUM_SCHEME_EXCHANGE_Q_MIRROR_GHOST,
      NUM_SCHEME_PRIM_UPDATE_PREDICTOR,
      NUM_SCHEME_SLOPES,
      NUM_SCHEME_COMPUTE_FLUXES,
      NUM_SCHEME_COMPUTE_VISCOUS_FLUXES,
      NUM_SCHEME_UPDATE,
      NUM_SCHEME_GRAVITY,
      NUM_SCHEME_ELEC_FIELD,
      NUM_SCHEME_SFACEMAG,
      NUM_SCHEME_COMPUTE_EMF,
      NUM_SCHEME_UPDATE_MAG,
      NUM_SCHEME_COMPUTE_BFACE_GHOSTED,
  IO
)
// clang-format on

class ProfilingTimer
{
private:
  static constexpr std::pair<const char *, Color_t>
  get_name_color(const ProfilingZone zone)
  {
    switch (zone)
    {
      // clang-format off
      ADD_CASE(AMR_CYCLE,                         "AMR_CYCLE                              ", FullBlue);
      ADD_CASE(AMR_CYCLE_RESIZE_AUX,              "AMR_CYCLE::RESIZE_AUX                  ", RoyalBlue);
      ADD_CASE(AMR_CYCLE_COPY_HASHMAP,            "AMR_CYCLE::COPY_HASHMAP                ", DodgerBlue);
      ADD_CASE(AMR_CYCLE_SYNC_MPI_GHOSTS,         "AMR_CYCLE::SYNC_MPI_GHOSTS             ", SteelBlue);
      ADD_CASE(AMR_CYCLE_MARK_CELLS,              "AMR_CYCLE::MARK_CELLS                  ", SteelBlue);
      ADD_CASE(AMR_CYCLE_ADAPT_MESH,              "AMR_CYCLE::ADAPT_MESH                  ", Lavender);
      ADD_CASE(AMR_CYCLE_USERDATA_REMAP,          "AMR_CYCLE::USERDATA_REMAP              ", DeepSkyBlue);
      ADD_CASE(LOAD_BALANCING,                    "LOAD_BALANCING                         ", LightBlue);
      ADD_CASE(LOAD_BALANCING_PARTITION_MESH,     "LOAD_BALANCING::PARTITION_MESH         ", LightBlue);
      ADD_CASE(LOAD_BALANCING_PARTITION_USERDATA, "LOAD_BALANCING::PARTITION_USERDATA     ", LightBlue);
      ADD_CASE(LOAD_BALANCING_UPDATE_MESH,        "LOAD_BALANCING::UPDATE_MESH            ", LightBlue);
      ADD_CASE(LOAD_BALANCING_RESIZE,             "LOAD_BALANCING::RESIZE                 ", LightBlue);
      ADD_CASE(NUM_CFL,                           "NUM::CFL                               ", DarkRed);
      ADD_CASE(NUM_SCHEME,                        "NUM::SCHEME                            ", FullRed);
      ADD_CASE(NUM_SCHEME_CONV_PRIM,              "NUM::SCHEME::CONV_PRIM                 ", LightRed);
      ADD_CASE(NUM_SCHEME_Q_GHOSTED,              "NUM::SCHEME::Q_GHOSTED                 ", LightRed);
      ADD_CASE(NUM_SCHEME_EXCHANGE_Q_MIRROR_GHOST,"NUM::SCHEME::EXCHANGE_Q_MIRROR_GHOST   ", LightRed);
      ADD_CASE(NUM_SCHEME_PRIM_UPDATE_PREDICTOR,  "NUM::SCHEME::PRIM_UPDATE_PREDICTOR     ", LightRed);
      ADD_CASE(NUM_SCHEME_SLOPES,                 "NUM::SCHEME::SLOPES                    ", Salmon);
      ADD_CASE(NUM_SCHEME_COMPUTE_FLUXES,         "NUM::SCHEME::COMPUTE_FLUXES            ", Coral);
      ADD_CASE(NUM_SCHEME_COMPUTE_VISCOUS_FLUXES, "NUM::SCHEME::COMPUTE_VISCOUS_FLUXES    ", Coral);
      ADD_CASE(NUM_SCHEME_UPDATE,                 "NUM::SCHEME::UPDATE                    ", Pink);
      ADD_CASE(NUM_SCHEME_GRAVITY,                "NUM::SCHEME::GRAVITY                   ", PaleGold);
      ADD_CASE(NUM_SCHEME_ELEC_FIELD,             "NUM::SCHEME::ELEC_FIELD                ", HotPink);
      ADD_CASE(NUM_SCHEME_SFACEMAG,               "NUM::SCHEME::SFACEMAG                  ", HotPink);
      ADD_CASE(NUM_SCHEME_COMPUTE_EMF,            "NUM::SCHEME::COMPUTE_EMF               ", Coral);
      ADD_CASE(NUM_SCHEME_UPDATE_MAG,             "NUM::SCHEME::UPDATE_MAG                ", Pink);
      ADD_CASE(NUM_SCHEME_COMPUTE_BFACE_GHOSTED,  "NUM::SCHEME::COMPUTE_BFACE_GHOSTED     ", DeepPurple);
      ADD_CASE(IO,                                "IO                                     ", FullBlue);
      // clang-format on
      default:
        return { zone._to_string(), Color_t::Default() };
    }
  }

public:
  static auto &
  get_profiling_region(ProfilingManager & profiling_mgr, const ProfilingZone zone)
  {
    const auto [name, color] = get_name_color(zone);
    return profiling_mgr.get_region(name, ProfilingRegion::TIMER_HOST, color);
  }

  ProfilingTimer(ProfilingManager & profiling_mgr, const ProfilingZone zone)
    : m_timer(get_profiling_region(profiling_mgr, zone))
  {
    m_timer.start();
  }

  ~ProfilingTimer() { m_timer.stop(); }

private:
  ProfilingRegion & m_timer;
};

} // namespace godunov_mhd_ct

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_MHD_PROFILING_DATA_H_
