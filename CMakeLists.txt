cmake_minimum_required(VERSION 3.26)
project(hiox)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

# onnxruntime
set(ONNXRUNTIME_ROOT D:/Software/onnxruntime-win-x64-1.18.0)
set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT}/include)
set(ONNXRUNTIME_LIBS ${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib)

include_directories(
        ${ONNXRUNTIME_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set(LIBS
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBS}
)

file(GLOB_RECURSE SRC
        src/*.cc
        src/*.cpp
        src/*.h
        src/*.hpp
)

add_executable(XFeatDemo XFeatDemo.cc ${SRC})
target_link_libraries(XFeatDemo ${LIBS})