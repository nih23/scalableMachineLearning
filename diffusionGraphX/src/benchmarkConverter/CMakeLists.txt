cmake_minimum_required(VERSION 2.4)

PROJECT(ReadBenchmark)

SET(primalLPBound_VERSION_MAJOR 1)
SET(primalLPBound_VERSION_MINOR 0)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -O2")
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules ${CMAKE_MODULE_PATH})
SET(SOURCES readBenchmark.cpp)

find_package(HDF5 REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenGM)

INCLUDE_DIRECTORIES(${OpenGM_INCLUDE_DIR}  
		    ${PROJECT_SOURCE_DIR}/include/
		    ${HDF5_INCLUDE_DIR})

ADD_EXECUTABLE(convBenchmark ${SOURCES})
TARGET_LINK_LIBRARIES(convBenchmark ${MPI_LIBRARIES} ${HDF5_LIBRARIES} rt)
