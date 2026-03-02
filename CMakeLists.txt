cmake_minimum_required(VERSION 3.10)
project(LlamaForge VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Compile options for strict production errors
add_compile_options(-Wall -Wextra -Werror -pedantic)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files (we'll just create stub cpp files to ensure the headers compile correctly)
set(SOURCES
    src/runtime/ModelStore.cpp
    src/session/SessionState.cpp
    src/runtime/InferenceContext.cpp
    src/runtime/ArenaAllocator.cpp
    src/runtime/PagedKVCache.cpp
)

# Core library
add_library(llamaforge_core STATIC ${SOURCES})
target_include_directories(llamaforge_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# We want to make sure the headers compile without syntax errors.

# Tests
add_executable(test_memory_managers tests/test_memory_managers.cpp)
target_link_libraries(test_memory_managers PRIVATE llamaforge_core)
