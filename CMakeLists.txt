cmake_minimum_required(VERSION 3.10)
project(ublarcvserver VERSION 0.1)

# set default install directory to
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/release" CACHE PATH "${CMAKE_BUILD_DIR}" FORCE )
endif()

## Git Submodules
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  # Update submodules as needed
  option(GIT_SUBMODULE "Check submodules during build" ON)
  if(GIT_SUBMODULE)
    message(STATUS "Submodule update")
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      RESULT_VARIABLE GIT_SUBMOD_RESULT)
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
      message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
    endif()
  endif()
endif()

## Dependencies

#---check majordomo library
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/majordomo/configure.ac")
  message(FATAL_ERROR "The json submodule was not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

# right now, we assume user built majordomo library
# probably should write a cmake file for majordomo
set(LIBMDP_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/release/libmdp")
set(LIBMDP_INC_DIR "${LIBMDP_INSTALL_DIR}/include")
set(LIBMDP_LIB_DIR "${LIBMDP_INSTALL_DIR}/lib")
set(LIBMDP_LIBRARIES "-L${LIBMDP_LIB_DIR} -lmajordomo")

#---CERN ROOT
find_package(ROOT CONFIG REQUIRED)
# --- Define useful ROOT functions and macros (e.g. ROOT_GENERATE_DICTIONARY)
include(${ROOT_USE_FILE})

#---LARCV
set( larcv_DIR $ENV{LARCV_BASEDIR}/build/installed/lib/cmake/larcv/)
find_package(larcv CONFIG REQUIRED)

# Set useful variables
set(UBLARCVSERVER_BASEDIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(UBLARCVSERVER_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/include")
set(UBLARCVSERVER_LIBRARY_DIR "${CMAKE_INSTALL_PREFIX}/lib")

# Add base of source
add_subdirectory(ublarcvserver)

#---Configuration export
include(CMakePackageConfigHelpers)
set(TARGETS_EXPORT_NAME "ublarcvserverTargets")
configure_package_config_file("cmake/Config.cmake.in"
  "${CMAKE_INSTALL_PREFIX}/share/ublarcvserver/ublarcvserverConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ublarcvserver"
  PATH_VARS UBLARCVSERVER_INCLUDE_DIR UBLARCVSERVER_LIBRARY_DIR )
write_basic_package_version_file("${CMAKE_INSTALL_PREFIX}/share/ublarcvserver/ublarcvserverConfigVersion.cmake"
  COMPATIBILITY SameMajorVersion)


## Export Targets

#---Create info to include ublarcvserver in other cmake files
install(EXPORT ublarcvserver
        FILE ublarcvserverTargets.cmake
        DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ublarcvserver")

#---Create info to include ublarcvserver in other cmake files
install(FILES "${CMAKE_INSTALL_PREFIX}/share/ublarcvserver/ublarcvserverConfig.cmake"
  "${CMAKE_INSTALL_PREFIX}/share/ublarcvserver/ublarcvserverConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ublarcvserver")
