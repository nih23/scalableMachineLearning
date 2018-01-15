cmake_minimum_required(VERSION 2.8)

project(diffusionmpi)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

add_executable(convertHDF5ToCSV convertHDF5ToCSV)
add_executable(acyclicGraph acyclicGraph)

find_package(OpenGM)
find_package(HDF5)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=gnu++11 ${OpenMP_CXX_FLAGS}")
set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS})
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS})

include_directories(${PROJECT_SOURCE_DIR}/include ${OpenGM_INCLUDE_DIR} ${HDF5_INCLUDE_DIR})

target_link_libraries(convertHDF5ToCSV hdf5)
target_link_libraries(acyclicGraph hdf5)
