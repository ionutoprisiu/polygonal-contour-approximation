cmake_minimum_required(VERSION 3.14)
project(PolygonalApproximation)

set(CMAKE_CXX_STANDARD 17)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(polygonal_approximation
        main.cpp
        src/lab11.cpp
        src/lab11.h
)

target_link_libraries(polygonal_approximation ${OpenCV_LIBS})

# Setăm output directory
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin) 