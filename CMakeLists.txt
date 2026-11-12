cmake_minimum_required(VERSION 3.16)
project(picospec VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(PICOSPEC_SANITIZE "Build with AddressSanitizer and UBSan" OFF)

add_library(picospec
  src/model.cpp
  src/cpu_features.cpp
  src/kernels_scalar.cpp)
target_include_directories(picospec PUBLIC include)
target_compile_options(picospec PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)

# The same reference source again, with the auto-vectoriser off. Benchmarks
# use it to show what the compiler contributes without any hand-written SIMD.
add_library(picospec_novec OBJECT src/kernels_scalar.cpp)
target_include_directories(picospec_novec PRIVATE include)
target_compile_definitions(picospec_novec PRIVATE PICOSPEC_VARIANT=novec)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(picospec_novec PRIVATE -fno-vectorize -fno-slp-vectorize)
else()
  target_compile_options(picospec_novec PRIVATE -fno-tree-vectorize)
endif()
target_sources(picospec PRIVATE $<TARGET_OBJECTS:picospec_novec>)

# Target architecture. On macOS a cross-build (for example
# -DCMAKE_OSX_ARCHITECTURES=x86_64 on Apple silicon) keeps the host's
# CMAKE_SYSTEM_PROCESSOR, so the requested architecture takes priority.
set(PICOSPEC_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_OSX_ARCHITECTURES)
  set(PICOSPEC_ARCH "${CMAKE_OSX_ARCHITECTURES}")
endif()
if(PICOSPEC_ARCH MATCHES "^(aarch64|arm64|ARM64)$")
  target_sources(picospec PRIVATE src/kernels_neon.cpp src/kernels_dotprod.cpp)
  # Only this one file may use SDOT; it is selected at run time.
  set_source_files_properties(src/kernels_dotprod.cpp PROPERTIES
    COMPILE_OPTIONS "-march=armv8.2-a+dotprod")
endif()

if(PICOSPEC_SANITIZE)
  foreach(t picospec picospec_novec)
    target_compile_options(${t} PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer -g)
  endforeach()
  target_link_options(picospec PUBLIC -fsanitize=address,undefined)
endif()

add_executable(picospec_cli tools/picospec_cli.cpp)
set_target_properties(picospec_cli PROPERTIES OUTPUT_NAME picospec)
target_link_libraries(picospec_cli PRIVATE picospec)

enable_testing()
add_executable(picospec_tests tests/test_main.cpp)
target_link_libraries(picospec_tests PRIVATE picospec)
add_test(NAME unit COMMAND picospec_tests ${CMAKE_SOURCE_DIR})
