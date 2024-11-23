hpcc_populate_dep(xxhash)

set(__XXHASH_SRC__ ${xxhash_SOURCE_DIR}/xxhash.c)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    list(APPEND __XXHASH_SRC__ ${xxhash_SOURCE_DIR}/xxh_x86dispatch.c)
endif()

add_library(xxhash_static STATIC ${__XXHASH_SRC__})
target_compile_definitions(xxhash_static PRIVATE DISPATCH=1)
target_include_directories(xxhash_static PUBLIC ${xxhash_SOURCE_DIR})
unset(__XXHASH_SRC__)
