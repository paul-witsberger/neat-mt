Notes for using CMAKE:
======================

*NOTE: run* ``restview cmake_notes.rst`` *from a command prompt to view this in a properly formatted version.*

#. Cmake *configures* the project, but it still needs to be *built* with another tool (e.g. Visual Studio)

#. Commands are case insensitive

#. A minimum working example::

	cmake_minimum_required(VERSION X.YY)
	project(ProjectName)
	add_executable(ProjectName project_name.cpp)

#. To build and test via command line::

	mkdir ProjectName_build
	cd ProjectName_build
	cmake ../ProjectName
	cmake --build .
	ProjectName

#. To add a library, we do the following:
	1. Assume we have a library to compute the square root function. There is a subdirectory ``MathFunctions`` that contains header file ``MathFunctions.h`` and source file ``mysqrt.cxx``. The source file has one function, ``mysqrt``.

	2. Add the following single line ``CMakeLists.txt`` file to the ``MathFunctions`` directory::

		add_library(MathFunctions mysqrt.cxx)

	3. Then use ``add_subdirectory()`` in the top-level ``CMakeLists.txt`` to build the library::

		add_subdirectory(MathFunctions)
		add_executable(ProjectName project_name.cpp)
		target_link_libraries(ProjectName PUBLIC MathFunctions)
		target_include_directories(ProjectName PUBLIC "${PROJECT_BINARY_DIR}" "${PROJECT_SOURCE_DIR}/MathFunctions")

#. Use ``--config Release`` (or ``--config Debug``) when building to specify configuration. So ``cmake --build .`` becomes ``cmake --build . --config Release``

#. Use ``-A x64`` (or ``-A Win32``) when compiling to specify architecture. So ``cmake ..`` becomes ``cmake -A x64 -S ..``.

#. Similar to the example above, we can make a static library with 1-2 lines. To build a .lib for the astrodynamics functions by Vallado, we run the following two lines::

	file(GLOB SOURCE ${CMAKE_SOURCE_DIR}/ast*.cpp)
	add_library(vallado STATIC ${SOURCE})

#. To compile ``test``, an executable which uses a static library:
	#. The project has directories:
		* ``src``: contains the main source file ``test_main.cpp``
		* ``test``: contains ``test.cpp``, ``test.h``, and ``CMakeLists.txt``. These files will be compiled into a static library that will be included and linked to the main file.
		* ``TestMain_build``: the location where the CMake files will be stored and is the location from which cmake should be called.
	#. Open a command prompt and navigate to ``TestMain_build``.
	#. ``cmake -A x64 -S ..``
	#. ``cmake --build . --config Release``
	#. The output ``TestMain.exe`` is located in ``TestMain_build\Release\``.
	#. The intermediate static library ``test.lib`` is located in ``TestMain_build\test\Release\``.

#. You can "install" files to a particular directory after building using ``cmake --build . --config Release --target install``.

#. How I compiled ``test_tbp_full.pyd``::

	cd C:\Users\pawit\Documents\codes\tbp_full_build
	cmake -A x64 -S ..
	cmake --build . --config Release --target install

#. The corresponding ``CMakeLists.txt`` is
	.. code-block:: cmake

		cmake_minimum_required(VERSION 3.12.0)
		PROJECT (tbp_full)

		set(PYTHON_VERSION 36)
		set(BOOST_ROOT "C:/Users/pawit/Documents/boost/boost_1_73_0")
		set(PYTHON_ROOT "C:/Program Files/Python36")
		set(VALLADO_ROOT "C:/Users/pawit/Documents/codes/src/vallado")

		set(BOOST_INCLUDE_DIR ${BOOST_ROOT})
		file(GLOB BOOST_LIBRARIES ${BOOST_ROOT}/stage/lib/*.lib)
		set(PYTHON_INCLUDE_DIR "${PYTHON_ROOT}/include")
		file(GLOB PYTHON_LIBRARIES ${PYTHON_ROOT}/libs/python*[^d].lib)
		set(VALLADO_INCLUDE_DIR ${VALLADO_ROOT})

		add_subdirectory(src/vallado)

		add_library(test_tbp_full SHARED src/test_full_tbp.cpp)
		target_include_directories(test_tbp_full PUBLIC ${VALLADO_INCLUDE_DIR} ${BOOST_INCLUDE_DIR} ${PYTHON_INCLUDE_DIR})
		target_link_libraries(test_tbp_full vallado ${BOOST_LIBRARIES} ${PYTHON_LIBRARIES})
		set_target_properties(test_tbp_full PROPERTIES SUFFIX ".pyd")

		install(TARGETS test_tbp_full DESTINATION ${CMAKE_SOURCE_DIR}/tbp_full_bin)
		install(DIRECTORY "${BOOST_ROOT}/stage/lib/" DESTINATION ${CMAKE_SOURCE_DIR}/tbp_full_bin FILES_MATCHING PATTERN "*.lib")
		install(DIRECTORY "${BOOST_ROOT}/stage/lib/" DESTINATION ${CMAKE_SOURCE_DIR}/tbp_full_bin FILES_MATCHING PATTERN "*.dll")
		install(DIRECTORY ${VALLADO_ROOT} DESTINATION ${CMAKE_SOURCE_DIR}/tbp_full_bin FILES_MATCHING PATTERN "*.lib")
