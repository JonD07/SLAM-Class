cmake_minimum_required(VERSION 3.1)
project(matcher)
set(CMAKE_CXX_STANDARD 17)

include_directories(include/)
include_directories(extern/eigen)

add_subdirectory(src)

