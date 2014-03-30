cmake_minimum_required(VERSION 2.8)

project(CUDACMAKETEST)

set(CMAKE_MODULE_PATH "./cmake" ${CMAKE_MODULE_PATH})

find_package(CUDA)



set(PY_INC "/usr/include/python2.7")
set(NUMPY_INC "/usr/lib/python2.7/dist-packages/numpy/core/include/numpy")

file(GLOB_RECURSE cpps "src/*.cpp")

file(GLOB_RECURSE headers "include/*.h")

file(GLOB_RECURSE cucpps "src/*.cu")

file(GLOB_RECURSE testsrc "test/*.cu")

file(GLOB_RECURSE cuheaders "include/*.cuh")


find_package(CBLAS REQUIRED)


set(COMMON_FLAGS "-DNUMPY_INTERFACE -DMODELNAME=_ConvNet -DINITNAME=init_ConvNet")

set(CUDA_NVCC_FLAGS ${COMMON_FLAGS};${CUDA_NVCC_FLAGS};-gencode arch=compute_35,code=sm_35)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_FLAGS}")

if (!CUDA_FOUND)
#	message("CUDA found, using device squaring!")
#	add_subdirectory(src.gpu)
	message("CUDA not found,")
#	add_subdirectory(src.cpu)
endif()

set ( BLAS_INCLUDE_DIRS "/usr/include" )

set ( INCLUDE_DIRS
 ${CBLAS_INCLUDE_DIR}
 ${PY_INC}
 ${NUMPY_INC}

 ${CMAKE_CURRENT_SOURCE_DIR}/include
 ${CMAKE_CURRENT_SOURCE_DIR}/include/common
 ${CMAKE_CURRENT_SOURCE_DIR}/include/nvmatrix
 ${CMAKE_CURRENT_SOURCE_DIR}/include/cudaconv2
 ${CMAKE_CURRENT_SOURCE_DIR}/sdk/common/inc
)

include_directories( ${INCLUDE_DIRS} )

link_directories( ${BLAS_LIBRARY_DIRS} )
CUDA_INCLUDE_DIRECTORIES( ${INCLUDE_DIRS} )
CUDA_ADD_LIBRARY( _ConvNet ${cpps} ${cucpps} SHARED )
CUDA_ADD_CUBLAS_TO_TARGET( _ConvNet )

set_target_properties(_ConvNet PROPERTIES PREFIX "")
target_link_libraries( _ConvNet ${CBLAS_LIBRARIES} )

CUDA_ADD_EXECUTABLE( test_conv ${testsrc} )
target_link_libraries( test_conv _ConvNet "python2.7")

#set(CMAKE_CXX_FLAG "-g -Wall")
