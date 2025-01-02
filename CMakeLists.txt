cmake_minimum_required(VERSION 3.10)
project(neural_network)

# Add the path where "sciplot/sciplot.hpp" lives
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/sciplot/sciplot")

add_executable(neural_network main.cpp)
