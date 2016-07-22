cmake_minimum_required(VERSION 2.8)

project(diffusionmpi)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)
add_subdirectory(src)
add_subdirectory(header)

#add_executable(diffmpi_old src/diffusionMPI)
add_executable(diffmpi src/diffusion_mpi_rewrite)
find_package(Boost COMPONENTS serialization mpi system)
find_package(MPI)
find_package(HDF5)
find_package(OpenGM)

#set(OPENGM_PATH /Users/nico/Code/opengm)
#set(OPENGM_PATH_BUILD ${OPENGM_PATH}/build)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=gnu++0x ${OpenMP_CXX_FLAGS}")
set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS})
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS})

## MPI
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
#include_directories(${MPI_INCLUDE_PATH} ${HDF5_INCLUDE_DIR} ${OPENGM_PATH}/include ${OPENGM_PATH}/src/external/QPBO-v1.3.src-patched)
include_directories(${Boost_INCLUDE_DIRS} ${MPI_INCLUDE_PATH} ${OpenGM_INCLUDE_DIR})
#target_link_libraries(diffmpi ${MPI_LIBRARIES} ${OPENGM_PATH})

## Serial
include_directories(${HDF5_INCLUDE_DIR} ${OpenGM_INCLUDE_DIR} ${OpenGM_INCLUDE_DIR}/../src/external/QPBO-v1.3.src-patched)

target_link_libraries(diffmpi hdf5 ${Boost_LIBRARIES} ${MPI_LIBRARIES})
#target_link_libraries(diffmpi_old ${OPENGM_PATH})
