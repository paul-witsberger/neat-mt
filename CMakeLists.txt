# NOTE: run the following commands from C:\Users\pawit\Documents\codes\boost_tbp_build
# NOTE: 	cmake -A x64 C:\Users\pawit\OneDrive\Documents\Classes\Research\neat-mt
# NOTE:		cmake --build . --config Release --target install

# Set up
cmake_minimum_required(VERSION 3.20.0)
PROJECT (boost_tbp)

set(PYTHON_VERSION 38)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MAKE_CXX_FLAGS} -std=c++0x -g -fPIC")

# Specify relevant paths
set(BOOST_ROOT /home/paul/Documents/boost/boost_1_73_0)
set(PYTHON_ROOT /usr/bin/python3.8)
set(VALLADO_ROOT /home/paul/Documents/build/vallado)
set(LOCAL_PYTHON_ENV /home/paul/.virtualenvs/research_env)
set(SOURCE_DIR /home/paul/Documents/neat-mt)

# Define Include and Link libraries
set(BOOST_INCLUDE_DIR ${BOOST_ROOT})
set(PYTHON_INCLUDE_DIR /usr/include/python3.8)
set(VALLADO_INCLUDE_DIR ${VALLADO_ROOT})

file(GLOB BOOST_LIBRARIES ${BOOST_ROOT}/stage/lib/*.so)
file(GLOB PYTHON_LIBRARIES /usr/lib/python3.8/config-3.8-x86_64-linux-gnu/*python*3*8*.so)

# Build Vallado
add_subdirectory(${VALLADO_ROOT} ${CMAKE_BINARY_DIR}/vallado)

# Create the main shared library
add_library(boost_tbp SHARED ${SOURCE_DIR}/boost_tbp.cpp)
target_include_directories(boost_tbp PUBLIC ${VALLADO_INCLUDE_DIR} ${BOOST_INCLUDE_DIR} ${PYTHON_INCLUDE_DIR})
target_link_libraries(boost_tbp vallado ${BOOST_LIBRARIES} ${PYTHON_LIBRARIES})

# Copy final file to source directory
install(TARGETS boost_tbp DESTINATION ${SOURCE_DIR})
