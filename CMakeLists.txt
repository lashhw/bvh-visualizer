cmake_minimum_required(VERSION 3.23)
project(bvh_visualizer)

set(CMAKE_CXX_STANDARD 17)

add_executable(bvh_visualizer main.cpp)
add_subdirectory(bvh)
target_link_libraries(bvh_visualizer PUBLIC bvh)
