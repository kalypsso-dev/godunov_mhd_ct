#
# Kalypsso MHD-CT solver
#
set(EXE_NAME "solver_godunov_mhd_ct")
add_executable(${EXE_NAME} solver_godunov_mhd_ct.cpp)

target_include_directories(${EXE_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src
                                              ${CMAKE_BINARY_DIR}/src)

target_link_libraries(${EXE_NAME} PRIVATE kalypsso_warnings)

target_link_libraries(
  ${EXE_NAME}
  PRIVATE kalypsso::core
          kalypsso::log
          kalypsso::p4est_wrap
          kalypsso::config
          kalypsso::godunov_mhd_ct
          Kokkos::kokkos
          kalypsso::monitoring)

if(KALYPSSO_CORE_USE_CNPY)
  target_link_libraries(${EXE_NAME} PRIVATE cnpy::cnpy)
endif()

if(KALYPSSO_CORE_USE_CPPTRACE)
  target_link_libraries(${EXE_NAME} PRIVATE cpptrace::cpptrace)
endif(KALYPSSO_CORE_USE_CPPTRACE)

add_subdirectory(brio-wu-2d)
add_subdirectory(brio-wu-3d)
add_subdirectory(dai-woodward)
add_subdirectory(slow_rarefaction)
add_subdirectory(expansion)

configure_file(test_mhd_blast_2d.ini test_mhd_blast_2d.ini COPYONLY)
configure_file(test_mhd_blast_2d_restart.ini test_mhd_blast_2d_restart.ini COPYONLY)
configure_file(test_mhd_blast_2d_balsara_spicer_1999.ini test_mhd_blast_2d_balsara_spicer_1999.ini
               COPYONLY)
configure_file(test_mhd_blast_2d_stone_gardiner_2008.ini test_mhd_blast_2d_stone_gardiner_2008.ini
               COPYONLY)
configure_file(test_mhd_blast_2d_caddy_schneider_2024.ini
               test_mhd_blast_2d_caddy_schneider_2024.ini COPYONLY)
configure_file(test_mhd_blast_3d.ini test_mhd_blast_3d.ini COPYONLY)
configure_file(test_mhd_diamag_cavity_2d.ini test_mhd_diamag_cavity_2d.ini COPYONLY)
configure_file(test_mhd_diamag_cavity_3d.ini test_mhd_diamag_cavity_3d.ini COPYONLY)
configure_file(test_mhd_field_loop_advection_2d.ini test_mhd_field_loop_advection_2d.ini COPYONLY)
configure_file(test_mhd_kelvin_helmholtz_2d.ini test_mhd_kelvin_helmholtz_2d.ini COPYONLY)
configure_file(test_mhd_kelvin_helmholtz_3d.ini test_mhd_kelvin_helmholtz_3d.ini COPYONLY)
# configure_file(test_mhd_kelvin_helmholtz_2d_stone_small.ini
# test_mhd_kelvin_helmholtz_2d_stone_small.ini COPYONLY)
# configure_file(test_mhd_kelvin_helmholtz_2d_stone_large.ini
# test_mhd_kelvin_helmholtz_2d_stone_large.ini COPYONLY)
configure_file(test_mhd_orszag_tang_2d.ini test_mhd_orszag_tang_2d.ini COPYONLY)
configure_file(test_mhd_orszag_tang_3d.ini test_mhd_orszag_tang_3d.ini COPYONLY)
configure_file(test_mhd_rayleigh_taylor_2d.ini test_mhd_rayleigh_taylor_2d.ini COPYONLY)
configure_file(test_mhd_rayleigh_taylor_3d.ini test_mhd_rayleigh_taylor_3d.ini COPYONLY)
configure_file(test_mhd_rotor_2d.ini test_mhd_rotor_2d.ini COPYONLY)

# block size study
add_subdirectory(block_size_study)
