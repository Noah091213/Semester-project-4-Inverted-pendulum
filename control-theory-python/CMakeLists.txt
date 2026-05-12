cmake_minimum_required(VERSION 3.28.0)
project(DataCollectionSystem)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(run main.cpp open62541.c)

target_link_libraries(run PRIVATE pthread)