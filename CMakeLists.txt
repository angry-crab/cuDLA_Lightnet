cmake_minimum_required(VERSION 3.0.0)
enable_language(CUDA)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(OpenCV REQUIRED)
find_package (gflags REQUIRED)
include_directories(/usr/local/cuda/include include)
link_directories(/usr/local/cuda/lib64 /usr/lib/aarch64-linux-gnu/nvidia/ /usr/local/cuda-12.2/targets/aarch64-linux/lib)
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

add_compile_options(-D LIGHTNET_STANDALONE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -Wall ${OpenMP_CXX_FLAGS} -D USE_DLA_STANDALONE_MODE")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O0 -g -Wno-write-strings ${OpenMP_CXX_FLAGS}" -d USE_DLA_STANDALONE_MODE)


include_directories (${gflags_INCLUDE_DIR} ./include ./include/nvsci_headers)

add_executable(cudla_lightnet src/main.cc src/config_parser.cc  src/cudla_context_standalone.cc src/cudla_lightnet.cc)
target_include_directories(cudla_lightnet PRIVATE  ${OpenCV_INCLUDE_DIRS})
target_link_libraries(cudla_lightnet ${OpenCV_LIBS} gflags  boost_system  boost_filesystem z cuda cudart nvinfer nvscibuf nvscisync nvcudla cudla "stdc++fs" "stdc++" "-ldl")
