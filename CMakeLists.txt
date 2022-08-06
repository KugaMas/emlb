cmake_minimum_required(VERSION 3.5.1)
project(EMLB)

# onnxruntime
# set(CMAKE_CXX_FLAGS "-std=c++11 -O3")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(CMAKE_BUILD_TYPE "Release")
set(Torch_DIR ${PROJECT_SOURCE_DIR}/extern/libtorch/share/cmake/Torch)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/scripts/utils)

aux_source_directory(${PROJECT_SOURCE_DIR}/scripts/src DIR_SRCS)

#最后发布加入这句话，编译用release模式，比默认的debug模式速度慢很多 
find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED CONFIG NO_DEFAULT_PATH)

# head file include
include_directories(${PROJECT_SOURCE_DIR}/include)
add_subdirectory(${PROJECT_SOURCE_DIR}/extern/pybind11)

pybind11_add_module(cdn_utils ${DIR_SRCS})

target_link_libraries(cdn_utils PRIVATE ${ONNXRUNTIME_LIB} ${TORCH_LIBRARIES})
