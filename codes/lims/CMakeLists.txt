cmake_minimum_required(VERSION 3.28)
project(lims)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD_REQUIRED ON)


set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g  -ggdb -fno-omit-frame-pointer -fno-inline-small-functions")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -g   -DNDEBUG -march=native -funroll-loops -ftree-vectorize -faligned-new -ffast-math")


file(GLOB_RECURSE LIMS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
add_library(LIMS STATIC ${LIMS_SRC})
set_target_properties(LIMS PROPERTIES LINKER_LANGUAGE CXX)


add_executable(run_lims run_lims.cpp)
target_link_libraries(run_lims LIMS)

