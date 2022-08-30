cmake_minimum_required(VERSION 3.5.1)
project(EMLB)

# onnxruntime
set(CMAKE_CXX_FLAGS                " ${CMAKE_CXX_FLAGS_INIT} -std=c++14 -fPIC")
set(CMAKE_CXX_FLAGS_DEBUG          "-g")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE        "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(CMAKE_BUILD_TYPE "RELEASE")
set(Torch_DIR ${PROJECT_SOURCE_DIR}/extern/libtorch/share/cmake/Torch)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/scripts/utils)

aux_source_directory(${PROJECT_SOURCE_DIR}/scripts/src DIR_SRCS)

find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)
find_package(Torch REQUIRED CONFIG NO_DEFAULT_PATH)

# head file include
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
add_subdirectory(${PROJECT_SOURCE_DIR}/extern/pybind11)

pybind11_add_module(cdn_utils ${DIR_SRCS})

target_link_libraries(cdn_utils PRIVATE ${TORCH_LIBRARIES})
