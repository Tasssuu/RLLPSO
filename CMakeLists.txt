cmake_minimum_required(VERSION 3.0)
project(LLSO)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -O4")

include(GNUInstallDirs)
set(CMAKE_CXX_STANDARD 17)
set(PACKAGE_NAME matplotlib_cpp)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/${PACKAGE_NAME}/cmake)


# Library target
add_library(matplotlib_cpp INTERFACE)
target_include_directories(matplotlib_cpp
    INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/examples>
        $<INSTALL_INTERFACE:include>
)
target_compile_features(matplotlib_cpp INTERFACE
    cxx_std_11
)
# TODO: Use `Development.Embed` component when requiring cmake >= 3.18
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
target_link_libraries(matplotlib_cpp INTERFACE
    Python3::Python
    Python3::Module
)
find_package(Python3 COMPONENTS NumPy)
if(Python3_NumPy_FOUND)
    target_link_libraries(matplotlib_cpp INTERFACE
        Python3::NumPy
    )
else()
    target_compile_definitions(matplotlib_cpp INTERFACE WITHOUT_NUMPY)
endif()
install(
    TARGETS matplotlib_cpp
    EXPORT install_targets
)

# ソースファイルを指定
set(SOURCE_FILES
        LLSO.cpp
        Self_Define_Functions.cpp
        ./CEC2010/Benchmarks.cpp
        ./CEC2010/F1.cpp
        ./CEC2010/F2.cpp
        ./CEC2010/F3.cpp
        ./CEC2010/F4.cpp
        ./CEC2010/F5.cpp
        ./CEC2010/F6.cpp
        ./CEC2010/F7.cpp
        ./CEC2010/F8.cpp
        ./CEC2010/F9.cpp
        ./CEC2010/F10.cpp
        ./CEC2010/F11.cpp
        ./CEC2010/F12.cpp
        ./CEC2010/F13.cpp
        ./CEC2010/F14.cpp
        ./CEC2010/F15.cpp
        ./CEC2010/F16.cpp
        ./CEC2010/F17.cpp
        ./CEC2010/F18.cpp
        ./CEC2010/F19.cpp
        ./CEC2010/F20.cpp
        )

# 実行可能ファイルを生成
add_executable(LLSO ${SOURCE_FILES})

target_link_libraries(LLSO PRIVATE matplotlib_cpp)
set_target_properties(LLSO PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")