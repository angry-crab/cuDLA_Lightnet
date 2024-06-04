# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

CUDA_PATH ?= /usr/local/cuda
BUILD_DIR := ./build
SRCDIR := ./src

CXX := g++
NVCC := $(CUDA_PATH)/bin/nvcc

USE_DLA_STANDALONE_MODE := 1

ALL_CCFLAGS += --std=c++17 -Wno-deprecated-declarations -Wall -fopenmp

ifeq ($(USE_DLA_STANDALONE_MODE),1)
    ALL_CCFLAGS += -DUSE_DLA_STANDALONE_MODE
	ifeq ($(USE_DETERMINISTIC_SEMAPHORE),1)
		ALL_CCFLAGS += -DUSE_DETERMINISTIC_SEMAPHORE
	endif
endif

ifeq ($(DEBUG),1)
    ALL_CCFLAGS += -g
else
    ALL_CCFLAGS += -O3
endif

NVCC_FLAGS := -gencode arch=compute_87,code=sm_87

OPENCV_INCLUDE_PATH ?= /usr/include/opencv4/
OPENCV_LIB_PATH ?= /usr/lib/aarch64-linux-gnu/

INCLUDES += -I $(CUDA_PATH)/include \
            -I $(OPENCV_INCLUDE_PATH) \
			-I /usr/include \
			-I /home/autoware/develop/nvsci_headers \
			-I /home/autoware/develop/cuDLA_Lightnet/include
LIBRARIES += -l cudla -L$(CUDA_PATH)/lib64 \
             -l cuda -l cudart -l nvinfer \
             -L $(OPENCV_LIB_PATH) \
			 -l pthread \
			 -l gflags \
	         -l opencv_objdetect -l opencv_highgui -l opencv_imgproc -l opencv_core -l opencv_imgcodecs -l opencv_dnn \
			 -L /usr/lib/aarch64-linux-gnu/nvidia/ -lnvscibuf \
			 -L /usr/lib/aarch64-linux-gnu/nvidia/ -lnvscisync

CXXSRCS := $(wildcard $(SRCDIR)/*.cc)
CXXOBJS := $(patsubst %.cc,$(BUILD_DIR)/%.o,$(notdir $(CXXSRCS)))
NVCCSRCS := $(wildcard $(SRCDIR)/*.cu)
NVCCOBJS := $(patsubst %.cu,$(BUILD_DIR)/%.o,$(notdir $(NVCCSRCS)))
all: cudla_lightnet

$(BUILD_DIR)/%.o: $(SRCDIR)/%.cc | $(BUILD_DIR)
	$(CXX) $(INCLUDES) $(ALL_CCFLAGS) -c -o $@ $<
    @echo "Compiled cxx object file: $@ from $<"

$(BUILD_DIR)/%.o: $(SRCDIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(INCLUDES) $(NVCC_FLAGS) -c -o $@ $<
    @echo "Compiled nvcc object file: $@"

cudla_lightnet: $(NVCCOBJS) $(CXXOBJS) | $(BUILD_DIR)
	$(CXX) $(ALL_CCFLAGS) $(INCLUDES) $(ALL_LDFLAGS) -o $(BUILD_DIR)/$@ $+ $(LIBRARIES)

run: cudla_lightnet
	./$(BUILD_DIR)/cudla_lightnet --engine=./model/loadable/lightnet.int8.fp16chwin.fp16chwout.standalone.bin --rgb=./config/T4x.colormap --names=./config/T4x.names --precision=fp16 --anchors=10,14,22,22,15,49,35,36,56,52,38,106,92,73,114,118,102,264,201,143,272,232,415,278,274,476,522,616,968,730 --num_anchors=5 --c=10 --subnet_engine=./model/loadable/lightnet_320.int8.fp16chwin.fp16chwout.standalone.bin --subnet_rgb=./config/anonymization.colormap --subnet_names=./config/anonymization.names --subnet_anchors=44,27,136,30,79,52,171,41,157,72,229,49,218,77,191,128,290,182 --subnet_num_anchors=3 --target_names=./config/personal_info.names --bluron=./config/personal_info.names --d=./data --save_detections_path=./results

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -f $(BUILD_DIR)/*

# export LD_LIBRARY_PATH=./src/matx_reformat/build/:$(OPENCV_LIB_PATH):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(OPENCV_LIB_PATH):$LD_LIBRARY_PATH