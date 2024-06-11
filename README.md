# cuDLA_Lightnet


# Env


Orin


Docker image (nvcr.io/nvidia/l4t-jetpack:r36.2.0)


# Dependencies


`sudo apt update`


`sudo apt install libopencv-dev python3-pip libgflags-dev git git-lfs`

For `nvsci_headers`, please you may get it from https://developer.nvidia.com/downloads/embedded/l4t/r36_release_v3.0/sources/public_sources.tbz2


# Instructions


`bash script/build_dla.sh`


`make -j`


`make run`


# Reference


https://github.com/tier4/trt-lightnet


https://github.com/angry-crab/cudla_dev

