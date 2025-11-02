cmake_minimum_required(VERSION 3.28)
project(pmtree)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD_REQUIRED ON)


set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g  -ggdb -fno-omit-frame-pointer -fno-inline-small-functions")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -g   -DNDEBUG -march=native -funroll-loops -ftree-vectorize -faligned-new -ffast-math")


file(GLOB_RECURSE PM_TREE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
add_library(PMTREE STATIC ${PM_TREE_SRC})
set_target_properties(PMTREE PROPERTIES LINKER_LANGUAGE CXX)


add_executable(run_pmtree run_pmtree.cpp)
target_link_libraries(run_pmtree PMTREE)


