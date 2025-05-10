cmake_minimum_required(VERSION 3.18)
project(cudbscan LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architecture
# You may need to adjust this for your specific GPU
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)

# Build options
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_JAVA_BINDINGS "Build Java bindings" ON)
option(BUILD_EXAMPLES "Build examples" ON)

# Find packages
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Python bindings
if(BUILD_PYTHON_BINDINGS)
  # Find pybind11 package
  find_package(pybind11 REQUIRED)
  
  # Set up the Python module
  pybind11_add_module(pycudbscan_core
    src/cuda/main.cu
    src/python/pybind_wrapper.cpp
  )
  
  # Link libraries
  target_link_libraries(pycudbscan_core PRIVATE ${CUDA_LIBRARIES})
  
  # Set output directory
  set_target_properties(pycudbscan_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/pycudbscan
  )
  
  # Install target for Python
  install(TARGETS pycudbscan_core
    LIBRARY DESTINATION python/pycudbscan
  )
  
  # Python package setup
  find_package(Python COMPONENTS Interpreter Development REQUIRED)
  
  # Install Python package using pip
  add_custom_target(python_install
    COMMAND ${Python_EXECUTABLE} -m pip install -e ${CMAKE_SOURCE_DIR}/python
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python
    COMMENT "Installing Python package with pip"
    DEPENDS pycudbscan_core
  )
endif()

# Java bindings
if(BUILD_JAVA_BINDINGS)
  # Find JNI package
  find_package(JNI)
  
  if(JNI_FOUND)
    # Include JNI directories
    include_directories(${JNI_INCLUDE_DIRS})
    
    # Create the JNI library
    add_library(jcudbscan_java SHARED
      src/cuda/main.cu
      src/java/java_wrapper.cpp
    )
    
    # Link libraries for JNI
    target_link_libraries(jcudbscan_java PRIVATE ${CUDA_LIBRARIES} ${JNI_LIBRARIES})
    
    # Set output directory for Java library
    set_target_properties(jcudbscan_java PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/java/lib
    )
    
    # Install target for Java
    install(TARGETS jcudbscan_java
      LIBRARY DESTINATION java/lib
    )
    
    # Find Java for building Java classes
    find_package(Java COMPONENTS Development)
    
    if(Java_FOUND)
      include(UseJava)
      
      # Set up Java compilation
      set(JAVA_SOURCE_FILES 
        ${CMAKE_SOURCE_DIR}/java/com/jcudbscan/JCUDBSCANJava.java
      )
      
      set(CMAKE_JAVA_COMPILE_FLAGS -d ${CMAKE_SOURCE_DIR}/java/build/classes)
      
      # Compile Java classes
      add_custom_target(java_compile
        COMMAND ${Java_JAVAC_EXECUTABLE} ${CMAKE_JAVA_COMPILE_FLAGS} ${JAVA_SOURCE_FILES}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Compiling Java classes"
        DEPENDS jcudbscan_java
      )
      
      # Create JAR file
      add_custom_target(java_jar
        COMMAND ${Java_JAR_EXECUTABLE} cf ${CMAKE_SOURCE_DIR}/java/build/jcudbscan.jar -C ${CMAKE_SOURCE_DIR}/java/build/classes .
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Creating JAR file"
        DEPENDS java_compile
      )
      
      add_custom_target(java_build DEPENDS java_jar)
    else()
      message(STATUS "Java compiler not found. Java classes will not be compiled.")
    endif()
  else()
    message(STATUS "JNI not found. Java bindings will not be built.")
    set(BUILD_JAVA_BINDINGS OFF)
  endif()
endif()

# Examples
if(BUILD_EXAMPLES)
  if(BUILD_PYTHON_BINDINGS)
    # Python example
    add_custom_target(run_python_example
      COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/examples/python/example.py
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      COMMENT "Running Python example"
      DEPENDS python_install
    )
  endif()
  
  if(BUILD_JAVA_BINDINGS AND Java_FOUND)
    # Java example
    add_custom_target(run_java_example
      COMMAND ${Java_JAVAC_EXECUTABLE} -cp ${CMAKE_SOURCE_DIR}/java/build/jcudbscan.jar 
              -d ${CMAKE_SOURCE_DIR}/examples/java/classes 
              ${CMAKE_SOURCE_DIR}/examples/java/ExampleJava.java
      COMMAND ${Java_JAVA_EXECUTABLE} -Djava.library.path=${CMAKE_SOURCE_DIR}/java/lib 
              -cp ${CMAKE_SOURCE_DIR}/java/build/jcudbscan.jar:${CMAKE_SOURCE_DIR}/examples/java/classes 
              ExampleJava
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      COMMENT "Running Java example"
      DEPENDS java_build
    )
  endif()
endif()

# Build all target
add_custom_target(build_all)

if(BUILD_PYTHON_BINDINGS)
  add_dependencies(build_all python_install)
endif()

if(BUILD_JAVA_BINDINGS AND JNI_FOUND AND Java_FOUND)
  add_dependencies(build_all java_build)
endif()

# Print build configuration
message(STATUS "Build configuration:")
message(STATUS "  Python bindings: ${BUILD_PYTHON_BINDINGS}")
message(STATUS "  Java bindings: ${BUILD_JAVA_BINDINGS}")
message(STATUS "  Examples: ${BUILD_EXAMPLES}")