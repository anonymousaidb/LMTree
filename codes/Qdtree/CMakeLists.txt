cmake_minimum_required(VERSION 3.28)
project(qdtree)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD_REQUIRED ON)


set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g  -ggdb -fno-omit-frame-pointer -fno-inline-small-functions")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -g   -DNDEBUG -march=native -funroll-loops -ftree-vectorize -faligned-new -ffast-math")


add_executable(run_qdtree main.cpp)
add_executable(set_radius set_radius.cpp)