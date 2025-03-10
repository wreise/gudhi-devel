# This files manage third party libraries required by GUDHI

find_package(Boost 1.66.0 QUIET OPTIONAL_COMPONENTS filesystem unit_test_framework program_options)

# Boost_FOUND is not reliable
if(NOT Boost_VERSION)
  message(FATAL_ERROR "NOTICE: This program requires Boost and will not be compiled.")
endif(NOT Boost_VERSION)
include_directories(${Boost_INCLUDE_DIRS})
message(STATUS "boost include dirs:" ${Boost_INCLUDE_DIRS})
message(STATUS "boost library dirs:" ${Boost_LIBRARY_DIRS})

find_package(GMP)
if(GMP_FOUND)
  INCLUDE_DIRECTORIES(${GMP_INCLUDE_DIR})
  find_package(GMPXX)
  if(GMPXX_FOUND)
    INCLUDE_DIRECTORIES(${GMPXX_INCLUDE_DIR})
  endif()
endif()

# from windows vcpkg eigen 3.4.0#2 : build fails with
# error C2440: '<function-style-cast>': cannot convert from 'Eigen::EigenBase<Derived>::Index' to '__gmp_expr<mpq_t,mpq_t>'
# cf. https://gitlab.com/libeigen/eigen/-/issues/2476
# Workaround is to compile with '-DEIGEN_DEFAULT_DENSE_INDEX_TYPE=int'
if (FORCE_EIGEN_DEFAULT_DENSE_INDEX_TYPE_TO_INT)
  message("++ User explicit demand to force EIGEN_DEFAULT_DENSE_INDEX_TYPE to int")
  add_definitions(-DEIGEN_DEFAULT_DENSE_INDEX_TYPE=int)
endif()

# In CMakeLists.txt, when include(${CGAL_USE_FILE}), CMAKE_CXX_FLAGS are overwritten.
# cf. http://doc.cgal.org/latest/Manual/installation.html#title40
# A workaround is to include(${CGAL_USE_FILE}) before adding "-std=c++11".
# A fix would be to use https://cmake.org/cmake/help/v3.1/prop_gbl/CMAKE_CXX_KNOWN_FEATURES.html
# or even better https://cmake.org/cmake/help/v3.1/variable/CMAKE_CXX_STANDARD.html
# but it implies to use cmake version 3.1 at least.
find_package(CGAL QUIET)

# Only CGAL versions > 4.11 supports what Gudhi uses from CGAL
if (CGAL_FOUND AND CGAL_VERSION VERSION_LESS 4.11.0)
  message("++ CGAL version ${CGAL_VERSION} is considered too old to be used by Gudhi.")
  unset(CGAL_FOUND)
  unset(CGAL_VERSION)
endif()

if(CGAL_FOUND)
  message(STATUS "CGAL version: ${CGAL_VERSION}.")
  include( ${CGAL_USE_FILE} )
endif()

option(WITH_GUDHI_USE_TBB "Build with Intel TBB parallelization" ON)

# Find TBB package for parallel sort - not mandatory, just optional.
if(WITH_GUDHI_USE_TBB)
  set(TBB_FIND_QUIETLY ON)
  find_package(TBB)
  if (TBB_FOUND)
    include(${TBB_USE_FILE})
    message("TBB found in ${TBB_LIBRARY_DIRS}")
    add_definitions(-DGUDHI_USE_TBB)
  endif()
endif(WITH_GUDHI_USE_TBB)

set(CGAL_WITH_EIGEN3_VERSION 0.0.0)
find_package(Eigen3 3.1.0)
if (EIGEN3_FOUND)
  include( ${EIGEN3_USE_FILE} )
  set(CGAL_WITH_EIGEN3_VERSION ${CGAL_VERSION})
endif (EIGEN3_FOUND)

# Required programs for unitary tests purpose
FIND_PROGRAM( GCOVR_PATH gcovr )
if (GCOVR_PATH)
  message("gcovr found in ${GCOVR_PATH}")
endif()
FIND_PROGRAM( GPROF_PATH gprof )
if (GPROF_PATH)
  message("gprof found in ${GPROF_PATH}")
endif()
FIND_PROGRAM( DIFF_PATH diff )
if (DIFF_PATH)
  message("diff found in ${DIFF_PATH}")
endif()
FIND_PROGRAM( GNUPLOT_PATH gnuplot )
if (GNUPLOT_PATH)
  message("gnuplot found in ${GNUPLOT_PATH}")
endif()

# BOOST ISSUE result_of vs C++11
add_definitions(-DBOOST_RESULT_OF_USE_DECLTYPE)
# BOOST ISSUE with Libraries name resolution under Windows
add_definitions(-DBOOST_ALL_NO_LIB)
# problem with Visual Studio link on Boost program_options
add_definitions( -DBOOST_ALL_DYN_LINK )
# problem on Mac with boost_system and boost_thread
add_definitions( -DBOOST_SYSTEM_NO_DEPRECATED )

if (WITH_GUDHI_PYTHON)
  # Find the correct Python interpreter.
  # Can be set with -DPYTHON_EXECUTABLE=/usr/bin/python3 or -DPython_ADDITIONAL_VERSIONS=3 for instance.
  find_package( PythonInterp )
  
  # find_python_module tries to import module in Python interpreter and to retrieve its version number
  # returns ${PYTHON_MODULE_NAME_UP}_VERSION and ${PYTHON_MODULE_NAME_UP}_FOUND
  function( find_python_module PYTHON_MODULE_NAME )
    string(TOUPPER ${PYTHON_MODULE_NAME} PYTHON_MODULE_NAME_UP)
    execute_process(
            COMMAND ${PYTHON_EXECUTABLE}  -c "import ${PYTHON_MODULE_NAME}; print(${PYTHON_MODULE_NAME}.__version__)"
            RESULT_VARIABLE PYTHON_MODULE_RESULT
            OUTPUT_VARIABLE PYTHON_MODULE_VERSION
            ERROR_VARIABLE PYTHON_MODULE_ERROR)
    if(PYTHON_MODULE_RESULT EQUAL 0)
      # Remove all carriage returns as it can be multiline
      string(REGEX REPLACE "\n" " " PYTHON_MODULE_VERSION "${PYTHON_MODULE_VERSION}")
      message ("++ Python module ${PYTHON_MODULE_NAME} - Version ${PYTHON_MODULE_VERSION} found")
  
      set(${PYTHON_MODULE_NAME_UP}_VERSION ${PYTHON_MODULE_VERSION} PARENT_SCOPE)
      set(${PYTHON_MODULE_NAME_UP}_FOUND TRUE PARENT_SCOPE)
    else()
      message ("PYTHON_MODULE_NAME = ${PYTHON_MODULE_NAME}
       - PYTHON_MODULE_RESULT = ${PYTHON_MODULE_RESULT}
       - PYTHON_MODULE_VERSION = ${PYTHON_MODULE_VERSION}
       - PYTHON_MODULE_ERROR = ${PYTHON_MODULE_ERROR}")
      unset(${PYTHON_MODULE_NAME_UP}_VERSION PARENT_SCOPE)
      set(${PYTHON_MODULE_NAME_UP}_FOUND FALSE PARENT_SCOPE)
    endif()
  endfunction( find_python_module )

  # For modules that do not define module.__version__
  function( find_python_module_no_version PYTHON_MODULE_NAME )
    string(TOUPPER ${PYTHON_MODULE_NAME} PYTHON_MODULE_NAME_UP)
    execute_process(
            COMMAND ${PYTHON_EXECUTABLE}  -c "import ${PYTHON_MODULE_NAME}"
            RESULT_VARIABLE PYTHON_MODULE_RESULT
            ERROR_VARIABLE PYTHON_MODULE_ERROR)
    if(PYTHON_MODULE_RESULT EQUAL 0)
      # Remove carriage return
      message ("++ Python module ${PYTHON_MODULE_NAME} found")
      set(${PYTHON_MODULE_NAME_UP}_FOUND TRUE PARENT_SCOPE)
    else()
      message ("PYTHON_MODULE_NAME = ${PYTHON_MODULE_NAME}
       - PYTHON_MODULE_RESULT = ${PYTHON_MODULE_RESULT}
       - PYTHON_MODULE_ERROR = ${PYTHON_MODULE_ERROR}")
      set(${PYTHON_MODULE_NAME_UP}_FOUND FALSE PARENT_SCOPE)
    endif()
  endfunction( find_python_module_no_version )
  
  if( PYTHONINTERP_FOUND )
    find_python_module("cython")
    find_python_module("pytest")
    find_python_module("matplotlib")
    find_python_module("numpy")
    find_python_module("scipy")
    find_python_module("sphinx")
    find_python_module("sklearn")
    find_python_module("ot")
    find_python_module("pybind11")
    find_python_module("torch")
    find_python_module("pykeops")
    find_python_module("eagerpy")
    find_python_module_no_version("hnswlib")
    find_python_module("tensorflow")
    find_python_module("sphinx_paramlinks")
    find_python_module("pydata_sphinx_theme")
  endif()
  
  if(NOT GUDHI_PYTHON_PATH)
    message(FATAL_ERROR "ERROR: GUDHI_PYTHON_PATH is not valid.")
  endif(NOT GUDHI_PYTHON_PATH)
  
  option(WITH_GUDHI_PYTHON_RUNTIME_LIBRARY_DIRS "Build with setting runtime_library_dirs. Useful when setting rpath is not allowed" ON)
  
  if(PYTHONINTERP_FOUND AND CYTHON_FOUND)
    if(SPHINX_FOUND)
      # Documentation generation is available through sphinx
      #find_program( SPHINX_PATH sphinx-build )
      # Calling sphinx-build may use a different version of python and fail
      set(SPHINX_PATH "${PYTHON_EXECUTABLE}" "-m" "sphinx.cmd.build")
    endif(SPHINX_FOUND)
  endif(PYTHONINTERP_FOUND AND CYTHON_FOUND)
endif (WITH_GUDHI_PYTHON)
