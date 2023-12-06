cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 14)
project(shape_based_matching)


# debug or release
SET(CMAKE_BUILD_TYPE "Release")


# arm or x86
IF(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
    SET(PLATFORM_COMPILE_FLAGS "-mfpu=neon")
ELSE()
    SET(PLATFORM_COMPILE_FLAGS "-march=native")

    # some places of the algorithm are designed for 128 SIMD
    # so 128 SSE may slightly faster than 256 AVX, you may want this
#    SET(PLATFORM_COMPILE_FLAGS "-msse -msse2 -msse3 -msse4 -mssse3")  # SSE only
ENDIF()

# SET(PLATFORM_COMPILE_FLAGS "-DMIPP_NO_INTRINSICS")  # close SIMD
SET(COMMON_COMPILE_FLAGS "-fopenmp -Wall -Wno-sign-compare -fPIC")
SET(CMAKE_CXX_FLAGS "${PLATFORM_COMPILE_FLAGS} ${COMMON_COMPILE_FLAGS} $ENV{CXXFLAGS}")
SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3")

# add CSV parser/writer utility
add_subdirectory(csv-parser)

#add Persistence layer
add_subdirectory(Persistence)

# opencv
# set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} /opt/ros/kinetic)
find_package(OpenCV 4 REQUIRED)
find_package(Boost 1.7 REQUIRED program_options)
find_package(Qt5 COMPONENTS Core REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${Qt5Core_INCLUDE_DIRS} Persistence/include csv-parser/include)

# include MIPP headers
include_directories (${INCLUDE_DIRECTORIES} "${CMAKE_CURRENT_SOURCE_DIR}/MIPP/")

# The linking of libPersistence.a with Qt libraries has already been done.
add_executable(
    ${PROJECT_NAME}_test
    nms.hpp
    utils.hpp utils.cpp
    line2Dup.h line2Dup.cpp
    test_jabil.cpp
)
target_link_libraries(${PROJECT_NAME}_test ${OpenCV_LIBS} ${Boost_LIBRARIES} Persistence stdc++fs)

add_executable(
    ${PROJECT_NAME}_test_old
    utils.hpp utils.cpp
    line2Dup.h line2Dup.cpp
    test_old.cpp
)
target_link_libraries(${PROJECT_NAME}_test_old ${OpenCV_LIBS} ${Boost_LIBRARIES} Persistence stdc++fs csv)

# add_executable(
#     SSIM
#     SSIM.cpp
# )
# target_link_libraries(SSIM ${OpenCV_LIBS} stdc++fs)

add_executable(
    test_scalrot
    test_scalrot.cpp
)
target_link_libraries(test_scalrot ${OpenCV_LIBS} ${Boost_LIBRARIES} Persistence stdc++fs)

file(COPY model_images DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
