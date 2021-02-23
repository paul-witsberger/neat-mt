# NOTE: run the following commands from C:\Users\pawit\Documents\codes\boost_tbp_build
# NOTE: 	cmake -A x64 C:\Users\pawit\OneDrive\Documents\Classes\Research\neat-mt
# NOTE:		cmake --build . --config Release --target install

# Set up
cmake_minimum_required(VERSION 3.20.0)
PROJECT (boost_tbp)

# Specify relevant paths
set(PYTHON_VERSION 38)
set(BOOST_ROOT "C:/Users/Paul/Documents/codes/boost_1_73_0")
set(PYTHON_ROOT "C:/Program Files/Python38")
set(VALLADO_ROOT "C:/Users/Paul/Documents/codes/src/vallado")
set(BIN_ROOT "C:/Users/Paul/Documents/codes/boost_tbp_bin")
set(LOCAL_PYTHON_ENV "C:/Users/Paul/Documents/research_env/Lib/site-packages/boost_tbp")
set(SOURCE_DIR "C:/Users/Paul/OneDrive/Documents/Classes/Research/neat-mt")

# Define Include and Link libraries
set(BOOST_INCLUDE_DIR ${BOOST_ROOT})
file(GLOB BOOST_LIBRARIES ${BOOST_ROOT}/stage/lib/*.lib)
set(PYTHON_INCLUDE_DIR "${PYTHON_ROOT}/include")
file(GLOB PYTHON_LIBRARIES ${PYTHON_ROOT}/libs/python*[^d].lib)
set(VALLADO_INCLUDE_DIR ${VALLADO_ROOT})

# Build Vallado
add_subdirectory(${VALLADO_ROOT} ${CMAKE_BINARY_DIR}/src/vallado)

# Create the main shared library
add_library(boost_tbp SHARED ${SOURCE_DIR}/boost_tbp.cpp)
target_include_directories(boost_tbp PUBLIC ${VALLADO_INCLUDE_DIR} ${BOOST_INCLUDE_DIR} ${PYTHON_INCLUDE_DIR})
target_link_libraries(boost_tbp vallado ${BOOST_LIBRARIES} ${PYTHON_LIBRARIES})
set_target_properties(boost_tbp PROPERTIES SUFFIX ".pyd")

# Copy relevant files to a convenient spot nearby
install(TARGETS boost_tbp DESTINATION ${BIN_ROOT})
install(DIRECTORY "${BOOST_ROOT}/stage/lib/" DESTINATION ${BIN_ROOT} FILES_MATCHING PATTERN "*.dll")
install(DIRECTORY "${BOOST_ROOT}/stage/lib/" DESTINATION ${BIN_ROOT} FILES_MATCHING PATTERN "*.lib")
install(DIRECTORY ${CMAKE_BINARY_DIR}/src/vallado DESTINATION ${BIN_ROOT} FILES_MATCHING PATTERN "*.lib")

# Copy relevant files to Python installation
install(TARGETS boost_tbp DESTINATION ${LOCAL_PYTHON_ENV})
install(DIRECTORY "${BOOST_ROOT}/stage/lib/" DESTINATION ${LOCAL_PYTHON_ENV} FILES_MATCHING PATTERN "*.dll")
install(DIRECTORY "${BOOST_ROOT}/stage/lib/" DESTINATION ${LOCAL_PYTHON_ENV} FILES_MATCHING PATTERN "*.lib")
install(DIRECTORY "${CMAKE_BINARY_DIR}/src/vallado/Release/" DESTINATION ${LOCAL_PYTHON_ENV} FILES_MATCHING PATTERN "*.lib")