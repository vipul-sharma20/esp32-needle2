idf_component_register(
    SRCS "main.c"
         "${CMAKE_CURRENT_LIST_DIR}/../../src/needle.c"
         "${CMAKE_CURRENT_LIST_DIR}/../../src/needle_cq.c"
         "${CMAKE_CURRENT_LIST_DIR}/../../src/needle_tok.c"
    INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/../../src"
    REQUIRES spi_flash esp_partition esp_timer driver vfs)
target_compile_options(${COMPONENT_LIB} PRIVATE -O3 -funroll-loops -fno-math-errno)
