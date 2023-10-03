include(FetchContent)
include(GoogleTest)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
  FetchContent_Populate(googletest)
  # Avoid installing gtest libs/headers by default
  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# gtest/aoclfftz_core_wrapper.h and gtest/aoclfftz_core_wrapper.c files
# will be added to a separately built core library for gtest
# since it uses the core functions directly to create wrapper version of
# functions with __declspec(dllexport) attribute
file(GLOB CORE_WRAPPER_FILES ${CMAKE_SOURCE_DIR}/gtest/aoclfftz_core_wrapper.h
                             ${CMAKE_SOURCE_DIR}/gtest/aoclfftz_core_wrapper.c)

# library will be generated in ${CMAKE_BINARY_DIR}/gtest
set(gtest_lib_name ${lib_name}_gtest)
if (BUILD_STATIC_LIBS)
    add_library(${gtest_lib_name} STATIC ${SRC_FILES} ${CORE_WRAPPER_FILES})
else ()
    add_library(${gtest_lib_name} SHARED ${SRC_FILES} ${CORE_WRAPPER_FILES})
endif()

file(GLOB GTEST_SRC_FILES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/gtest/*/*.cpp)

foreach(FILE ${GTEST_SRC_FILES})
    string(REGEX MATCH "[A-Za-z0-9_]*_gtest" TNAME ${FILE})
    message(${FILE})
    add_executable(${TNAME} ${FILE})
    target_link_libraries(${TNAME} ${gtest_lib_name} GTest::gtest_main)
endforeach()

foreach(FILE ${GTEST_SRC_FILES})
    string(REGEX MATCH "[A-Za-z0-9_]*_gtest" TNAME ${FILE})
    gtest_discover_tests(${TNAME})
endforeach()