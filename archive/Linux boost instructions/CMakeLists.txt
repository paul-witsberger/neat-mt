# From /home/roger/aaerg13/Witsberger/boost/boost_tbp, run: ./build.sh

cmake_minimum_required(VERSION 2.8.12)
PROJECT (boost_tbp)

# Specify relevant paths
set(PYTHON_VERSION 36)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MAKE_CXX_FLAGS} -std=c++0x -g -fPIC")
set(SOURCE_DIR "${CMAKE_SOURCE_DIR}/src")
set(BOOST_ROOT "/home/roger/aaerg13/Witsberger/boost/boost_1_69_0")
set(PYTHON_ROOT "/home/roger/aaerg13/Witsberger/anaconda3/envs/tbp/")
set(VALLADO_ROOT "/home/roger/aaerg13/Witsberger/Software/vallado")
set(BIN_ROOT "/home/roger/aaerg13/Witsberger/NN/Evolutionary")

# Set Include directories
set(BOOST_INCLUDE_DIR ${BOOST_ROOT})
set(PYTHON_INCLUDE_DIR "${PYTHON_ROOT}/include/python3.6m")
set(VALLADO_INCLUDE_DIR ${VALLADO_ROOT})

# Set Link Libraries
file(GLOB BOOST_LIBRARIES ${BOOST_ROOT}/stage/lib/*.so)
file(GLOB PYTHON_LIBRARIES ${PYTHON_ROOT}/lib/libpython*.so)

add_subdirectory(${VALLADO_ROOT} ${CMAKE_BINARY_DIR}/vallado)

# Build and link the python library module for each project
add_library(boost_tbp SHARED ${SOURCE_DIR}/boost_tbp.cpp)
target_include_directories(boost_tbp PUBLIC ${VALLADO_INCLUDE_DIR} ${BOOST_INCLUDE_DIR} ${PYTHON_INCLUDE_DIR})
target_link_libraries(boost_tbp vallado ${BOOST_LIBRARIES} ${PYTHON_LIBRARIES})