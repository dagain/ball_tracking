cmake_minimum_required(VERSION 3.10.0)
project(ball_tracking VERSION 0.1.0 LANGUAGES C CXX)

add_executable(ball_tracking main.cpp)

include(CTest)
enable_testing()



find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS})

target_link_libraries( ball_tracking ${OpenCV_LIBS})


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
