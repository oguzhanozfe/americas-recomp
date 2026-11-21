cmake_minimum_required(VERSION 3.20)
project(xbox_tools CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

get_filename_component(XBOX_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(XBOX_REC "${XBOX_ROOT}/core/shader/recompiler")

file(READ "${XBOX_REC}/CMakeLists.txt" XBOX_REC_CMAKE)
string(REGEX MATCHALL "\\\${CMAKE_CURRENT_FUNCTION_LIST_DIR}/[^ \t\r\n\")]+\\.cpp" XBOX_REC_MATCHES "${XBOX_REC_CMAKE}")
set(XBOX_REC_SOURCES "")
foreach(m IN LISTS XBOX_REC_MATCHES)
    string(REPLACE "\${CMAKE_CURRENT_FUNCTION_LIST_DIR}/" "" rel "${m}")
    if(rel MATCHES "DummyShaders|SpirvOptimizer|^tests/|/tests/")
        continue()
    endif()
    list(APPEND XBOX_REC_SOURCES "${XBOX_REC}/${rel}")
endforeach()
list(REMOVE_DUPLICATES XBOX_REC_SOURCES)

file(GLOB XBOX_REC_INC1 LIST_DIRECTORIES true "${XBOX_REC}/*/include")
file(GLOB XBOX_REC_INC2 LIST_DIRECTORIES true "${XBOX_REC}/*/*/include")

add_library(xbox_recompiler STATIC ${XBOX_REC_SOURCES} "${CMAKE_CURRENT_SOURCE_DIR}/stub_version.cpp")
target_include_directories(xbox_recompiler PUBLIC
    "${XBOX_ROOT}/core/libs"
    "${XBOX_ROOT}/3rdparty/SPIRV-Headers/include"
    "${XBOX_REC}"
    ${XBOX_REC_INC1} ${XBOX_REC_INC2})
target_compile_options(xbox_recompiler PUBLIC -w)
if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
    target_include_directories(xbox_recompiler PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/msvc")
    target_compile_definitions(xbox_recompiler PUBLIC NOMINMAX)
    target_compile_options(xbox_recompiler PUBLIC /EHsc)
endif()
find_package(Threads REQUIRED)
target_link_libraries(xbox_recompiler PUBLIC Threads::Threads)

add_executable(driver driver.cpp)
target_link_libraries(driver PRIVATE xbox_recompiler)
add_executable(stages stages.cpp)
target_link_libraries(stages PRIVATE xbox_recompiler)

set(XBOX_AGC "${XBOX_ROOT}/core/libs/prx/libSceAgcDriver/Graphics/src")
add_executable(exec exec.cpp
    "${XBOX_AGC}/GuestSamplerResource.cpp"
    "${XBOX_AGC}/GuestTextureResource.cpp"
    "${XBOX_AGC}/RuntimeSwitches.cpp"
    "${XBOX_AGC}/TextureFormat.cpp"
    "${XBOX_AGC}/TextureTiling.cpp"
    "${XBOX_ROOT}/core/libs/prx/libc/src/RunReport.cpp")
target_include_directories(exec PRIVATE "${XBOX_ROOT}/3rdparty/Vulkan-Headers/include" "${XBOX_ROOT}/core")
target_link_libraries(exec PRIVATE xbox_recompiler ${CMAKE_DL_LIBS})
