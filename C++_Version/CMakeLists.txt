cmake_minimum_required(VERSION 3.6)
project(Embedding)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES Embedding.cpp networks.h networks.tpp)
add_executable(Embedding.cpp ${SOURCE_FILES})