/* 
 * Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * 
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef COMMON_EXTRA_CUH
#define	COMMON_EXTRA_CUH

#include <helper_cuda.h>
#include <nvmatrix.cuh>
#include "conv_util.cuh"


void convMaxPoolFilterActs(NVMatrix& images, NVMatrix& filters, NVMatrix& targetsVal, NVMatrix &targetsSwitch, 
			    int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
			    int numImgColors, int numGroups,
			    int mpSizeX, int mpStart, int mpStride, int mpOutputsX);

void convMaxpoolWeightActs(NVMatrix& images, NVMatrix& hidActs, NVMatrix& hidSwitches, NVMatrix& targets,
			   int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart, int moduleStride, int numImgColors, int numGroups, int partialSum,
			   int mpSizeX, int mpStart, int mpStride,  int mpOutputsX);

void convMaxpoolWeightActs(NVMatrix& images, NVMatrix& hidActs, NVMatrix &hidSwitches,  NVMatrix& targets,
			   int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart, int moduleStride, int numImgColors, int numGroups, int partialSum,
			   int mpSizeX, int mpStart, int mpStride,  int mpOutputsX, float scaleTargets, float scaleOutput);

void convMaxpoolWeightActs2(NVMatrix& images, NVMatrix& hidActs, NVMatrix& hidSwitches, NVMatrix& targets,
			   int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart, int moduleStride, int numImgColors, int numGroups, int partialSum,
			   int mpSizeX, int mpStart, int mpStride,  int mpOutputsX);

void convMaxpoolWeightActs2(NVMatrix& images, NVMatrix& hidActs, NVMatrix &hidSwitches,  NVMatrix& targets,
			   int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart, int moduleStride, int numImgColors, int numGroups, int partialSum,
			   int mpSizeX, int mpStart, int mpStride,  int mpOutputsX, float scaleTargets, float scaleOutput);

void convImgActs2(NVMatrix& hidActs, NVMatrix& hidSwitches, NVMatrix& filters, NVMatrix& targets,
		  int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups,
		  int mpSizeX, int mpStart, int mpStride,  int mpOutputsX);


void convImgActs2(NVMatrix& hidActs, NVMatrix& hidSwitches, NVMatrix& filters, NVMatrix& targets,
		 int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups,
		  int mpSizeX, int mpStart, int mpStride,  int mpOutputsX,
		  float scaleTargets, float scaleOutput);
#endif	/* COMMON_EXTRA_CUH */

