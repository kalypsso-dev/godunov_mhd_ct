set(GODUNOV_MHD_CT_SRCS
    border_conditions/FillOutside.cpp
    border_conditions/FillOutside.h
    common.h
    init/FixVectorPotentialAtNonConformalEdges.h
    init/init.cpp
    init/init.h
    init/InitBlast.cpp
    init/InitBlast.h
    init/InitDiamagCavity.cpp
    init/InitDiamagCavity.h
    init/InitFieldLoopAdvection.cpp
    init/InitFieldLoopAdvection.h
    init/InitKelvinHelmholtz.cpp
    init/InitKelvinHelmholtz.h
    init/InitOrszagTang.cpp
    init/InitOrszagTang.h
    init/InitRayleighTaylor.cpp
    init/InitRayleighTaylor.cpp
    init/InitRotor.cpp
    init/InitRotor.h
    init/InitShockTube.cpp
    init/InitShockTube.h
    models/HydroSettings.cpp
    models/HydroSettings.h
    models/MHD.cpp
    models/MHD.h
    models/MHDSettings.cpp
    models/MHDSettings.h
    models/MHDState.h
    models/mhd_utils.h
    models/riemann_solver_types.h
    models/RiemannSolvers_MHD.h
    profiling.h
    scheme/AddGravitySourceTerm.cpp
    scheme/AddGravitySourceTerm.h
    scheme/ComputeDtMHDFunctor.cpp
    scheme/ComputeDtMHDFunctor.h
    scheme/ComputeElectricFieldFunctor.cpp
    scheme/ComputeElectricFieldFunctor.h
    scheme/ComputeEmfAndStoreFunctor.cpp
    scheme/ComputeEmfAndStoreFunctor.h
    scheme/ComputeLimitedSlopesFunctor.cpp
    scheme/ComputeLimitedSlopesFunctor.h
    scheme/ComputeSourceFaceMagFunctor.cpp
    scheme/ComputeSourceFaceMagFunctor.h
    scheme/ComputeUpdatedPrimitiveVariablesFunctor.cpp
    scheme/ComputeUpdatedPrimitiveVariablesFunctor.h
    scheme/ComputeHydroFluxesAndStoreFunctor.cpp
    scheme/ComputeHydroFluxesAndStoreFunctor.h
    scheme/ComputeHydroFluxesAndUpdateFunctor.cpp
    scheme/ComputeHydroFluxesAndUpdateFunctor.h
    scheme/ComputeViscousFluxesAndStoreFunctor.cpp
    scheme/ComputeViscousFluxesAndStoreFunctor.h
    scheme/ConvertToPrimitivesVariablesFunctor.cpp
    scheme/ConvertToPrimitivesVariablesFunctor.h
    scheme/CorrectEmfAtBlockBorderFunctor.cpp
    scheme/CorrectEmfAtBlockBorderFunctor.h
    scheme/GodunovImplemBase.h
    scheme/GodunovImplemV0.cpp
    scheme/GodunovImplemV0.h
    scheme/ReadEmfAndUpdateFunctor.cpp
    scheme/ReadEmfAndUpdateFunctor.h
    scheme/ReadFluxesAndConservativeUpdateFunctor.cpp
    scheme/ReadFluxesAndConservativeUpdateFunctor.h
    SolverGodunovMHD.cpp
    SolverGodunovMHD.h
    utils/ComputeDerivedQuantities.h)

add_library(godunov_mhd_ct ${GODUNOV_MHD_CT_SRCS})
target_include_directories(godunov_mhd_ct PUBLIC ${CMAKE_SOURCE_DIR}/src
                                                 ${CMAKE_SOURCE_DIR}/external/better-enums)

target_link_libraries(godunov_mhd_ct PRIVATE kalypsso_warnings)

target_link_libraries(godunov_mhd_ct PUBLIC Kokkos::kokkos kalypsso::core kalypsso::mpiUtils
                                            kalypsso::p4est_wrap)

# declare library alias
add_library(kalypsso::godunov_mhd_ct ALIAS godunov_mhd_ct)

# application main
add_subdirectory(app)
