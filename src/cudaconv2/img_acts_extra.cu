#include <cudaconv2_extra.cuh>

/*
 * Block size: 16x16.
 * blockIdx.x determines case in batches of 16*imgsPerThread.
 * blockIdx.y determines 4x4 image region in target image.
 *
 * threadIdx.x determines case.
 * threadIdx.y determines pixel.
 *
 * hidActs:     (numFilters, numModulesY, numModulesX, numImages)
 * filters:     (numColors, filterPixels, numFilters)                               if conv
 *              (numModulesY, numModulesX, numColors, filterPixels, numFilters)     otherwise
 * targets:     (numColors, imgSizeY, imgSizeX, numImages)
 *
 * Each block reconstructs one 4x4 pixels from 16*imgsPerThread cases.
 *
 * Number of filters must be divisible by 16.
 * Number of images must be divisible by 16*imgsPerThread  if checkCaseBounds is false.
 * 16 * imgsPerThread must be divisible by 32.
 *
 * This version loads 32 cases at a time, so it gets full coalescing on that load.
 * It only loads 16 weights at a time, so those aren't fully coalesced.
 * This version conserves shared memory by loading 16 filters at a time rather than 32.
 */
template <int imgsPerThread, int numColors, bool scale, bool checkCaseBounds, bool conv>
__global__ void img_acts_color2(const float* hidActs, const float*_hidSwitches, const float* filters, float* targets,
                                   const int numModulesY, const int numModulesX, const int numImages, const int numFilters,
                                   const int filterSize, const int imgSizeY, const int imgSizeX,
                                   const int paddingStart, const int moduleStride,
				const int mpSizeX, const int mpStart, const int mpStride, const int mpOutputsX,
                                   const float scaleTargets, const float scaleOutputs) {
    __shared__ float shFilters[numColors*16][16 + 1];
    __shared__ float shHidActs[16][16*imgsPerThread];
    
    const int* hidSwitches = (const int *) _hidSwitches;

    const int blockCaseIdx = blockIdx.x * 16*imgsPerThread;
    const int numRegionsX = DIVUP(imgSizeX, 4);
    const int blockRegionIdx = blockIdx.y;
    const int blockRegionIdxX = blockRegionIdx % numRegionsX;
    const int blockRegionIdxY = blockRegionIdx / numRegionsX;
    const int blockRegionLeft = blockRegionIdxX * 4;
    const int blockRegionTop = blockRegionIdxY * 4;
    const int pxYInRegion = threadIdx.y / 4, pxXInRegion = threadIdx.y % 4;
    const int pxY = blockRegionTop + pxYInRegion;
    const int pxX = blockRegionLeft + pxXInRegion;
    const int pxIdx = pxY * imgSizeX + pxX;
    const bool isPxInImg = pxY < imgSizeY && pxX < imgSizeX;
    const int numModules = numModulesY * numModulesX;
    const int filterPixels = filterSize * filterSize;
    const int imgPixels = imgSizeX * imgSizeY;
    const int tidx = threadIdx.y * 16 + threadIdx.x;
    const int loadY = tidx / 32, loadX = tidx % 32;

    const int mp_numModules = mpOutputsX * mpOutputsX;

    hidActs += blockCaseIdx + loadY * numImages * mp_numModules + loadX;
    hidSwitches += blockCaseIdx + loadY * numImages * mp_numModules + loadX;

    filters += threadIdx.x;
    targets += pxIdx * numImages + blockCaseIdx + threadIdx.x;


    float prod[numColors][imgsPerThread];
    #pragma unroll
    for (int c = 0; c < numColors; c++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[c][i] = 0;
        }
    }
    const int startY = blockRegionTop - paddingStart < filterSize ? 0
                        : 1 + (blockRegionTop - paddingStart - filterSize) / moduleStride;
    const int endY = MIN(numModulesY, 1 + (blockRegionTop + 3 - paddingStart) / moduleStride);
    const int startX = blockRegionLeft - paddingStart < filterSize ? 0
                        : 1 + (blockRegionLeft - paddingStart - filterSize) / moduleStride;
    const int endX = MIN(numModulesX, 1 + (blockRegionLeft + 3 - paddingStart) / moduleStride);
    
    float* shilterLoad = &shFilters[threadIdx.y][threadIdx.x];
    float* shHidActLoad = &shHidActs[loadY][loadX];

    for (int my = startY; my < endY; my++) {
        const int moduleTop = paddingStart + my * moduleStride;
        const int pxInModuleY = pxY - moduleTop;

        for (int mx = startX; mx < endX; mx++) {
            const int moduleIdx = my * numModulesX + mx;
            const int moduleLeft = paddingStart + mx * moduleStride;
            const int pxInModuleX = pxX - moduleLeft;

            const bool isPxInModule = pxInModuleY >= 0 && pxInModuleY < filterSize && pxInModuleX >= 0 && pxInModuleX < filterSize;
            const int pxIdxInModule = pxInModuleY * filterSize + pxInModuleX;

	    const int convTargetY = moduleIdx  / numModulesX;
	    const int convTargetX = moduleIdx % numModulesX;

	    const int loopStartY = MAX(0, DIVUP(convTargetY - mpStart - (mpSizeX-1),  mpStride));
	    const int loopStartX = MAX(0, DIVUP(convTargetX - mpStart - (mpSizeX-1),  mpStride));
	    const int loopEndY = MIN(mpOutputsX-1, (convTargetY - mpStart) / mpStride);
	    const int loopEndX = MIN(mpOutputsX-1, (convTargetX - mpStart) / mpStride);

            for (int f = 0; f < numFilters; f += 16) { // multiply with 16 filters at a time
                // Now the threads split up into half-warps, and each half-warp decides if it's interested.
		//                const float* hLoad = &hidActs[(moduleIdx + f * numModules) * numImages];
		// to load the moduleIdx + f module

                #pragma unroll
                for (int i = 0; i < imgsPerThread * 16; i += 32) {
                    if (!checkCaseBounds || blockCaseIdx + i + loadX < numImages) {
                        #pragma unroll
                        for (int j = 0; j < 16; j += 8) { // load 16 rows of imgsPerThread*16 cols, 8 * 32 elements at a time.
			    shHidActLoad[j * 16 * imgsPerThread + i] = 0;
                             #pragma unroll
			    for (int y = loopStartY; y <= loopEndY; y++) {
                                #pragma unroll
				for (int x = loopStartX; x <= loopEndX; x++) {
				    const int mpOutIdx = (y * mpOutputsX + x) * numImages + i + (f+j) * numImages * mp_numModules;
				    
				    shHidActLoad[j * 16 * imgsPerThread + i] += hidSwitches[mpOutIdx] == moduleIdx? hidActs[mpOutIdx] : 0;
				    //hLoad[j * numModules * numImages + i];
				}
			    }
                        }
                    } else {
                        #pragma unroll
                        for (int j = 0; j < 16; j += 8) { // load 16 rows of imgsPerThread*16 cols, 8 * 32 elements at a time.
                            shHidActLoad[j * 16 * imgsPerThread + i] = 0;
                        }
                    }
                }
                
                if (isPxInImg && isPxInModule) {
                    // This half-warp is interested, so it's going to load the weights from this module to its pixel.
                    // Not fully coalesced read :(
                    // But taking out this read entirely only reduces the runtime by ~2.8%, so it isn't costing me much.
                    const float* fLoad = conv ? &filters[pxIdxInModule * numFilters + f]
                                              : &filters[(moduleIdx * numColors * filterPixels + pxIdxInModule) * numFilters + f];
                    #pragma unroll
                    for (int c = 0; c < numColors; c++) {
                        shilterLoad[c * 16 * (16 + 1)] = fLoad[c * filterPixels * numFilters];
                    }

                    
                }

                __syncthreads();
                // Do some actual computation
                if (isPxInImg && isPxInModule) {
                    #pragma unroll
                    for (int c = 0; c < numColors; c++) {
                        #pragma unroll
                        for (int w = 0; w < 16; w++) {
                            #pragma unroll
                            for (int i = 0; i < imgsPerThread; i++) {
                                prod[c][i] += shFilters[threadIdx.y + c * 16][w] * shHidActs[w][threadIdx.x + i * 16];
                            }
                        }
                    }
                }
                __syncthreads();
            }
        }
    }
    // Not fully coalesced write :(... shmem (and fully coalesced) version is actually slightly slower, though
    if (isPxInImg) {
        if (scale) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || blockCaseIdx + threadIdx.x + i * 16 < numImages) {
                    #pragma unroll
                    for (int c = 0; c < numColors; c++) {
                        targets[c * imgPixels * numImages + i * 16] = scaleTargets * targets[c * imgPixels * numImages + i * 16] + scaleOutputs * prod[c][i];
                    }
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || blockCaseIdx + threadIdx.x + i * 16 < numImages) {
                    #pragma unroll
                    for (int c = 0; c < numColors; c++) {
                        targets[c * imgPixels * numImages + i * 16] = scaleOutputs * prod[c][i];
                    }
                }
            }
        }
    }
}

/*
 * Block size: 16x16.
 * blockIdx.x determines case in batches of 16*imgsPerThread, also color in batches of colorsPerThread.
 *  In essence, blockIdx.x.x = 1..numImages/(16*imgsPerThread)
 *              blockIdx.x.y = 1..numImgColors/colorsPerThread
 * blockIdx.y determines 4x4 image region in target image.
 *
 * threadIdx.x determines case.
 * threadIdx.y determines pixel.
 *
 * hidActs:     (numFilters, numModulesY, numModulesX, numImages)
 * filters:     (numFilterColors, filterPixels, numFilters)                             if conv
 *              (numModulesY, numModulesX, numFilterColors, filterPixels, numFilters)   otherwise
 * targets:     (numImageColors, imgSizeY, imgSizeX, numImages)
 *
 * Each block reconstructs one 4x4 pixels from 16*imgsPerThread cases.
 *
 * numImages must be divisible by 16*imgsPerThread if checkCaseBounds is false.
 * 16 * imgsPerThread must be divisible by 32.
 * numImageColors/numGroups must be divisible by colorsPerThread.
 *
 * This version loads 32 cases at a time, so it gets full coalescing on that load.
 * It only loads 16 weights at a time, so those aren't fully coalesced.
 * This version conserves shared memory by loading 16 filters at a time rather than 32.
 * 
 * To be used when there are 4-16 color channels.
 */
template <int imgsPerThread, int colorsPerThread,  bool scale, bool checkCaseBounds, bool conv>
__global__ void img_acts_mediumcolor2(const float* hidActs, const float* _hidSwitches, const float* filters, float* targets,
				     const int numModulesY, const int numModulesX, const int numImages, const int numFilters,
				     const int filterSize, const int imgSizeY, const int imgSizeX, const int paddingStart,
				     const int moduleStride, const int numImgColors, const int numGroups,
				      const int mpSizeX, const int mpStart, const int mpStride, const int mpOutputsX,
				     const float scaleTargets, const float scaleOutputs) {
    __shared__ float shFilters[colorsPerThread*16][16 + 1];
    __shared__ float shHidActs[16][16*imgsPerThread];

    const int* hidSwitches = (const int*) _hidSwitches;
    
    const int numImgBlocks = DIVUP(numImages,16*imgsPerThread);
    const int blockCaseIdx = (blockIdx.x % numImgBlocks) * 16*imgsPerThread;

    const int imgColorIdx = (blockIdx.x / numImgBlocks) * colorsPerThread; // color idx globally
    const int numFilterColors = numImgColors / numGroups;
    const int blockGroupIdx = imgColorIdx / numFilterColors;
    const int filterColorIdx = imgColorIdx % numFilterColors; // color idx within group
    const int numFiltersPerGroup = numFilters / numGroups;
    const int blockFilterIdx = blockGroupIdx * numFiltersPerGroup;
    
    const int numRegionsX = DIVUP(imgSizeX, 4);
    const int blockRegionIdx = blockIdx.y;
    const int blockRegionIdxX = blockRegionIdx % numRegionsX;
    const int blockRegionIdxY = blockRegionIdx / numRegionsX;
    const int blockRegionLeft = blockRegionIdxX * 4;
    const int blockRegionTop = blockRegionIdxY * 4;
    const int pxYInRegion = threadIdx.y / 4, pxXInRegion = threadIdx.y % 4;
    const int pxY = blockRegionTop + pxYInRegion;
    const int pxX = blockRegionLeft + pxXInRegion;
    const int pxIdx = pxY * imgSizeX + pxX;
    const bool isPxInImg = pxY < imgSizeY && pxX < imgSizeX;
    const uint numModules = numModulesY * numModulesX;
    const int filterPixels = filterSize * filterSize;
    const int imgPixels = imgSizeY * imgSizeX;
    const int tidx = threadIdx.y * 16 + threadIdx.x;
    const int loadY = tidx / 32, loadX = tidx % 32;

    const int mp_numModules = mpOutputsX * mpOutputsX;
    
    hidActs += blockCaseIdx + (blockFilterIdx + loadY) * numImages * mp_numModules + loadX;
    hidSwitches += blockCaseIdx + (blockFilterIdx + loadY) * numImages * mp_numModules + loadX;

    filters += blockFilterIdx + filterColorIdx * filterPixels * numFilters + threadIdx.x;
    targets += imgColorIdx * imgPixels * numImages + pxIdx * numImages + blockCaseIdx + threadIdx.x;

    float prod[colorsPerThread][imgsPerThread];
    #pragma unroll
    for (int c = 0; c < colorsPerThread; c++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[c][i] = 0;
        }
    }
    const int startY = blockRegionTop - paddingStart < filterSize ? 0
	: 1 + (blockRegionTop - paddingStart - filterSize) / moduleStride;
    const int endY = MIN(numModulesY, 1 + (blockRegionTop + 3 - paddingStart) / moduleStride);
    const int startX = blockRegionLeft - paddingStart < filterSize ? 0
	: 1 + (blockRegionLeft - paddingStart - filterSize) / moduleStride;
    const int endX = MIN(numModulesX, 1 + (blockRegionLeft + 3 - paddingStart) / moduleStride);

    float* shFilterLoad = &shFilters[threadIdx.y][threadIdx.x];
    float* shHidActLoad = &shHidActs[loadY][loadX];

    for (int my = startY; my < endY; my++) {
        const int moduleTop = paddingStart + my * moduleStride;
        const int pxInModuleY = pxY - moduleTop;

        for (int mx = startX; mx < endX; mx++) {
            const int moduleIdx = my * numModulesX + mx;
            const int moduleLeft = paddingStart + mx * moduleStride;
            const int pxInModuleX = pxX - moduleLeft;

            const bool isPxInModule = pxInModuleY >= 0 && pxInModuleY < filterSize && pxInModuleX >= 0 && pxInModuleX < filterSize;
            const int pxIdxInModule = pxInModuleY * filterSize + pxInModuleX;

	    const int convTargetY = moduleIdx  / numModulesX;
	    const int convTargetX = moduleIdx % numModulesX;

	    const int loopStartY = MAX(0, DIVUP(convTargetY - mpStart - (mpSizeX-1),  mpStride));
	    const int loopStartX = MAX(0, DIVUP(convTargetX - mpStart - (mpSizeX-1),  mpStride));
	    const int loopEndY = MIN(mpOutputsX-1, (convTargetY - mpStart) / mpStride);
	    const int loopEndX = MIN(mpOutputsX-1, (convTargetX - mpStart) / mpStride);

            for (int f = 0; f < numFiltersPerGroup; f += 16) { // multipply with 16 filters at a time
                // Now the threads split up into half-warps, and each half-warp decides if it's interested.
                //const float* hLoad = &hidActs[(moduleIdx + f * numModules) * numImages];
                #pragma unroll
                for (int i = 0; i < imgsPerThread * 16; i += 32) {
                    if (!checkCaseBounds || blockCaseIdx + loadX + i < numImages) {
                        #pragma unroll
                        for (int j = 0; j < 16; j += 8) { // load 16 rows of imgsPerThread*16 cols, 8 * 32 elements at a time.

			    shHidActLoad[j * 16 * imgsPerThread + i] = 0;
                             #pragma unroll
			    for (int y = loopStartY; y <= loopEndY; y++) {
                                #pragma unroll
				for (int x = loopStartX; x <= loopEndX; x++) {
				    const int mpOutIdx = (y * mpOutputsX + x) * numImages + i + (f+j) * numImages * mp_numModules;
				    shHidActLoad[j * 16 * imgsPerThread + i] += hidSwitches[mpOutIdx] == moduleIdx? hidActs[mpOutIdx] : 0;
				}
			    }
			    //                            shHidActLoad[j * 16 * imgsPerThread + i] = hLoad[j * numModules * numImages + i];
                        }
                    } else {
                        #pragma unroll
                        for (int j = 0; j < 16; j += 8) { // load 16 rows of imgsPerThread*16 cols, 8 * 32 elements at a time.
                            shHidActLoad[j * 16 * imgsPerThread + i] = 0;
                        }
                    }
                }

                if (isPxInImg && isPxInModule) {
                    // This half-warp is interested, so it's going to load the weights from this module to its pixel.
         
                    // Not fully coalesced read :(
                    // But taking out this read entirely only reduces the runtime by ~2.8%, so it isn't costing me much.
                    const float* fLoad = conv ? &filters[pxIdxInModule * numFilters + f]
			: &filters[moduleIdx * numFilterColors * filterPixels * numFilters + pxIdxInModule * numFilters + f];
                    #pragma unroll
                    for (int c = 0; c < colorsPerThread; c++) {
                        shFilterLoad[c * 16 * (16 + 1)] = fLoad[c * filterPixels * numFilters];
                    }
                }

                __syncthreads();
                // Do some actual computation
                if (isPxInImg && isPxInModule) {
                    #pragma unroll
                    for (int c = 0; c < colorsPerThread; c++) {
                        #pragma unroll
                        for (int w = 0; w < 16; w++) {
                            #pragma unroll
                            for (int i = 0; i < imgsPerThread; i++) {
                                prod[c][i] += shFilters[threadIdx.y + c * 16][w] * shHidActs[w][threadIdx.x + i * 16];
                            }
                        }
                    }
                }
                __syncthreads();
            }
        }
    }
    // Not fully coalesced write :(... shmem (and fully coalesced) version is actually slightly slower, though
    if (isPxInImg) {
        if (scale) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || blockCaseIdx + threadIdx.x + i * 16 < numImages) {
                    #pragma unroll
                    for (int c = 0; c < colorsPerThread; c++) {
                        targets[c * imgPixels * numImages + i * 16] = scaleTargets * targets[c * imgPixels * numImages + i * 16] + scaleOutputs * prod[c][i];
                    }
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || blockCaseIdx + threadIdx.x + i * 16 < numImages) {
                    #pragma unroll
                    for (int c = 0; c < colorsPerThread; c++) {
                        targets[c * imgPixels * numImages + i * 16] = scaleOutputs * prod[c][i];
                    }
                }
            }
        }
    }
}


/*
 * Block size: B_YxB_X.
 * blockIdx.x determines case in batches of B_X*imgsPerThread, also color in batches of B_Y*colorsPerThread.
 *  In essence, blockIdx.x.x = 1..numImages/(B_X*imgsPerThread)
 *              blockIdx.x.y = 1..numImgColors/(B_Y*colorsPerThread)
 * blockIdx.y determines image pixel in target image.
 *
 * threadIdx.x determines case.
 * threadIdx.y determines color.
 *
 * hidActs:     (numFilters, numModulesY, numModulesX, numImages)
 * filters:     (numFilterColors, filterPixels, numFilters)                             if conv
 *              (numModulesY, numModulesX, numFilterColors, filterPixels, numFilters)   otherwise
 * targets:     (numImageColors, imgSizeY, imgSizeX, numImages)
 *
 * Each block reconstructs one B_Y*colorsPerThread colors from 1 pixel from B_X*imgsPerThread cases.
 *
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false.
 * numFiltersPerGroup must be divisible by 16.
 * 
 * B_X * imgsPerThread must be divisible by 32.
 * numFilterColors must be divisible by B_Y*colorsPerThread.
 * B_X*B_Y must be divisible by 32.
 *
 * This version loads 32 cases at a time, so it gets full coalescing on that load.
 * It only loads 16 weights at a time, so those aren't fully coalesced.
 * This version conserves shared memory by loading 16 filters at a time rather than 32.
 * 
 * To be used when there are >= 16 color channels.
 */
template <int B_Y, int B_X, int imgsPerThread, int colorsPerThread, bool scale, bool checkCaseBounds, bool conv>
__global__ void conv_img_acts_manycolor2(const float* hidActs, const float* _hidSwitches, const float* filters, float* targets,
					const int numModulesY, const int numModulesX, const int numImages, const int numFilters,
					const int filterSize, const int imgSizeY, const int imgSizeX, const int paddingStart, const int moduleStride,
					const int numImgColors, const int numGroups,
					 int mpSizeX, int mpStart, int mpStride,  int mpOutputsX,
					const float scaleTargets, const float scaleOutputs) {

    const int *hidSwitches = (const int *) _hidSwitches;
    __shared__ float shFilters[colorsPerThread*B_Y][16 + 1]; // TODO: perhaps reconsider this 16
    __shared__ float shHidActs[16][B_X*imgsPerThread];

    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int blockCaseIdx = (blockIdx.x % numImgBlocks) * B_X*imgsPerThread;
    
    const int imgColorIdx = (blockIdx.x / numImgBlocks) * B_Y*colorsPerThread; // color idx globally
    const int numFilterColors = numImgColors / numGroups;
    const int blockGroupIdx = imgColorIdx / numFilterColors;
    const int filterColorIdx = imgColorIdx % numFilterColors; // color idx within group
    const int numFiltersPerGroup = numFilters / numGroups;
    const int blockFilterIdx = blockGroupIdx * numFiltersPerGroup;

    const int blockPixelIdx = blockIdx.y;
    const int blockPixelIdxX = blockPixelIdx % imgSizeX;
    const int blockPixelIdxY = blockPixelIdx / imgSizeX;

    const int filterPixels = filterSize * filterSize;
    const int imgPixels = imgSizeY * imgSizeX;
    const int tidx = threadIdx.y * B_X + threadIdx.x;
    const int hidActLoadY = tidx / 32, hidActLoadX = tidx % 32;
    const int filtersLoadY = tidx / 16, filtersLoadX = tidx % 16;
    //    const int numModules = numModulesY * numModulesX;

    const int mp_numModules = mpOutputsX * mpOutputsX;
    
    hidActs     += blockCaseIdx + (blockFilterIdx + hidActLoadY) * numImages * mp_numModules + hidActLoadX;
    hidSwitches += blockCaseIdx + (blockFilterIdx + hidActLoadY) * numImages * mp_numModules + hidActLoadX;

    //    hidActs += blockCaseIdx + (blockFilterIdx + hidActLoadY) * numImages * numModules + hidActLoadX;
    filters += blockFilterIdx + (filterColorIdx + filtersLoadY) * filterPixels * numFilters + filtersLoadX;
    targets += (imgColorIdx + threadIdx.y) * imgPixels * numImages + blockPixelIdx * numImages + blockCaseIdx + threadIdx.x;

    float prod[colorsPerThread][imgsPerThread];
    #pragma unroll
    for (int c = 0; c < colorsPerThread; c++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[c][i] = 0;
        }
    }

    const int startY = blockPixelIdxY - paddingStart < filterSize ? 0
	: 1 + (blockPixelIdxY - paddingStart - filterSize) / moduleStride;
    const int endY = MIN(numModulesY, 1 + (blockPixelIdxY - paddingStart) / moduleStride);
    const int startX = blockPixelIdxX - paddingStart < filterSize ? 0
	: 1 + (blockPixelIdxX - paddingStart - filterSize) / moduleStride;
    const int endX = MIN(numModulesX, 1 + (blockPixelIdxX - paddingStart) / moduleStride);

    float* shFilterLoad = &shFilters[filtersLoadY][filtersLoadX];
    float* shHidActLoad = &shHidActs[hidActLoadY][hidActLoadX];

    for (int my = startY; my < endY; my++) {
        const int moduleTop = paddingStart + my * moduleStride;
        const int pxInFilterY = blockPixelIdxY - moduleTop;

        for (int mx = startX; mx < endX; mx++) {
            const int moduleIdx = my * numModulesX + mx;
            const int moduleLeft = paddingStart + mx * moduleStride;
            const int pxInFilterX = blockPixelIdxX - moduleLeft;
            
            const int pxIdxInFilter = pxInFilterY * filterSize + pxInFilterX;

	    const int convTargetY = moduleIdx  / numModulesX;
	    const int convTargetX = moduleIdx % numModulesX;

	    const int loopStartY = MAX(0, DIVUP(convTargetY - mpStart - (mpSizeX-1),  mpStride));
	    const int loopStartX = MAX(0, DIVUP(convTargetX - mpStart - (mpSizeX-1),  mpStride));
	    const int loopEndY = MIN(mpOutputsX-1, (convTargetY - mpStart) / mpStride);
	    const int loopEndX = MIN(mpOutputsX-1, (convTargetX - mpStart) / mpStride);

            for (int f = 0; f < numFiltersPerGroup; f += 16) { // multiply with 16 filters at a time
		//                const float* hLoad = &hidActs[(moduleIdx + f * numModules) * numImages];
                #pragma unroll
                for (int i = 0; i < imgsPerThread * B_X; i += 32) {
                    if (!checkCaseBounds || blockCaseIdx + hidActLoadX + i < numImages) {
                        #pragma unroll
                        for (int j = 0; j < 16; j += B_X*B_Y/32) { // load 16 rows of imgsPerThread*16 cols, 8 * 32 elements at a time.

			    shHidActLoad[j * B_X * imgsPerThread + i] = 0;
                             #pragma unroll
			    for (int y = loopStartY; y <= loopEndY; y++) {
                                #pragma unroll
				for (int x = loopStartX; x <= loopEndX; x++) {
				    const int mpOutIdx = (y * mpOutputsX + x) * numImages + i + (f+j) * numImages * mp_numModules;
				    shHidActLoad[j * B_X * imgsPerThread + i] += hidSwitches[mpOutIdx] == moduleIdx? hidActs[mpOutIdx] : 0;
				    //shHidActLoad[j * B_X * imgsPerThread + i] +=  hidSwitches[mpOutIdx];
				}
			    }

			    //                            shHidActLoad[j * B_X * imgsPerThread + i] = hLoad[j * numModules * numImages + i];
                        }
                    } else {
                        #pragma unroll
                        for (int j = 0; j < 16; j += B_X*B_Y/32) { // load 16 rows of imgsPerThread*16 cols, 8 * 32 elements at a time.
                            shHidActLoad[j * B_X * imgsPerThread + i] = 0;
                        }
                    }
                }
                const float* fLoad = conv ? &filters[pxIdxInFilter * numFilters + f]
		    : &filters[moduleIdx * numFilterColors * filterPixels * numFilters + pxIdxInFilter * numFilters + f];
                #pragma unroll
                for (int i = 0; i < colorsPerThread*B_Y; i+= B_X*B_Y/16) {
                    if ((colorsPerThread*B_Y) % (B_X*B_Y/16) == 0 || i + filtersLoadY < colorsPerThread*B_Y) {
                        shFilterLoad[i * (16 + 1)] = fLoad[i * filterPixels * numFilters];
                    }
                }
                
                __syncthreads();
                // Do some actual computation
                #pragma unroll
                for (int c = 0; c < colorsPerThread; c++) {
                    #pragma unroll
                    for (int w = 0; w < 16; w++) {
                        #pragma unroll
                        for (int i = 0; i < imgsPerThread; i++) {
                            prod[c][i] += shFilters[c * B_Y + threadIdx.y][w] * shHidActs[w][threadIdx.x + i * B_X];
                        }
                    }
                }
                __syncthreads();
            }
        }
    }
    if (scale) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || blockCaseIdx + threadIdx.x + i * B_X < numImages) {
                #pragma unroll
                for (int c = 0; c < colorsPerThread; c++) {
                    targets[c * B_Y * imgPixels * numImages + i * B_X] = scaleTargets * targets[c * B_Y * imgPixels * numImages + i * B_X] + scaleOutputs * prod[c][i];
                }
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || blockCaseIdx + threadIdx.x + i * B_X < numImages) {
                #pragma unroll
                for (int c = 0; c < colorsPerThread; c++) {
                    targets[c * B_Y * imgPixels * numImages + i * B_X] = scaleOutputs * prod[c][i];
                }
            }
        }
    }
}

void _imgActs2(NVMatrix& hidActs, NVMatrix& hidSwitches, NVMatrix& filters, NVMatrix& targets,
              int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups,
	      int mpSizeX, int mpStart, int mpStride,  int mpOutputsX,
              float scaleTargets, float scaleOutput, bool conv) {
    int numFilterColors = numImgColors / numGroups;
    int numImages = hidActs.getNumCols();
    int numFilters = filters.getNumCols();
    int numModules = numModulesY * numModulesY;
    int filterModuleMult = conv ? 1 : numModules;
    int filterPixels = filters.getNumRows() / (filterModuleMult * numFilterColors);
    int filterSize = sqrt(filterPixels);
    int imgPixels = imgSizeY * imgSizeX;
    int numModulesX = numModules / numModulesY;
    int numOutputs = mpOutputsX * mpOutputsX;

    assert(numImgColors % numGroups == 0);
    assert(numFilters % (16*numGroups) == 0);
    assert(numGroups > 1 || (numImgColors > 0 && (numImgColors <= 3 || numImgColors % 2 == 0)));
    assert(numGroups == 1 || numFilterColors % 4 == 0);

    assert(filterPixels == filterSize * filterSize);
    assert(hidActs.getNumRows() == numOutputs * numFilters);
    assert(filters.getNumRows() == filterModuleMult * numFilterColors * filterPixels);
    assert(numModules == numModulesY * numModulesX);

    assert(hidActs.isContiguous());
    assert(filters.isContiguous());

    assert(!hidActs.isTrans());
    assert(!filters.isTrans());
    assert(!targets.isTrans());
    // These routines don't handle the case when only part of the image is visited in the convolution
    assert(paddingStart <= 0);
    assert(paddingStart + (numModulesX-1)*moduleStride + filterSize >= imgSizeX);
    assert(paddingStart + (numModulesY-1)*moduleStride + filterSize >= imgSizeY);
    assert(moduleStride <= filterSize);
    
    assert(targets.isContiguous()); // no stride support here!

    dim3 blocks;
    dim3 threads(16,16);
    int colorsPerThread;
    int imgsPerThread = numImages % 128 == 0 ? 8 : numImages % 64 == 0 ? 4 : 2;
    if (numFilterColors % 8 == 0) {
        threads = dim3(32, 4);
        colorsPerThread = numFilterColors % 16 == 0 ? 4 : 2;
        imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
        assert(numFilterColors % (threads.y * colorsPerThread) == 0);
        
        blocks = dim3(DIVUP(numImages, threads.x*imgsPerThread) * (numImgColors/(threads.y*colorsPerThread)), imgPixels);
    } else if (numFilterColors > 3) {
        colorsPerThread = numFilterColors % 4 == 0 ? 4 : 2;
        blocks = dim3(DIVUP(numImages,threads.x*imgsPerThread) * (numImgColors / colorsPerThread), DIVUP(imgSizeY,4) * DIVUP(imgSizeX,4));
    } else {
        blocks = dim3(DIVUP(numImages,threads.x*imgsPerThread), DIVUP(imgSizeY,4) * DIVUP(imgSizeX,4));
    }
    bool checkCaseBounds = numImages % (threads.x * imgsPerThread) != 0;
    
    if (scaleTargets == 0) { // do not scale or use targets matrix
        targets.resize(numImgColors*imgPixels, numImages);
    } else {
        assert(targets.getNumRows() == numImgColors * imgPixels);
        assert(targets.getNumCols() == numImages);
    }



    if (conv) { // convolutional units
        if (scaleTargets == 0) { // do not scale or use targets matrix
            if (numFilterColors % 8 == 0) {
                if (imgsPerThread == 4) {
                    if (checkCaseBounds) {
                        if (numFilterColors % 16 == 0) {
                            cudaFuncSetCacheConfig(conv_img_acts_manycolor2<4, 32, 4, 4, false, true, true>, cudaFuncCachePreferShared);
                            conv_img_acts_manycolor2<4, 32, 4, 4, false, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
													 numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else {
                            cudaFuncSetCacheConfig(conv_img_acts_manycolor2<4, 32, 4, 2, false, true, true>, cudaFuncCachePreferShared);
                            conv_img_acts_manycolor2<4, 32, 4, 2, false, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
													 numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        }
                    } else {
                        if (numFilterColors % 16 == 0) {
                            cudaFuncSetCacheConfig(conv_img_acts_manycolor2<4, 32, 4, 4, false, false, true>, cudaFuncCachePreferShared);
                            conv_img_acts_manycolor2<4, 32, 4, 4, false, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
													  numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else {
                            cudaFuncSetCacheConfig(conv_img_acts_manycolor2<4, 32, 4, 2, false, false, true>, cudaFuncCachePreferShared);
                            conv_img_acts_manycolor2<4, 32, 4, 2, false, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
													  numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        }
                    }
                } else if (imgsPerThread == 2) {
                    if (checkCaseBounds) {
                        if (numFilterColors % 16 == 0) {
                            cudaFuncSetCacheConfig(conv_img_acts_manycolor2<4, 32, 2, 4, false, true, true>, cudaFuncCachePreferShared);
                            conv_img_acts_manycolor2<4, 32, 2, 4, false, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
													 numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else {
                            cudaFuncSetCacheConfig(conv_img_acts_manycolor2<4, 32, 2, 2, false, true, true>, cudaFuncCachePreferShared);
                            conv_img_acts_manycolor2<4, 32, 2, 2, false, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
													 numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        }
                    } else {
                        if (numFilterColors % 16 == 0) {
                            cudaFuncSetCacheConfig(conv_img_acts_manycolor2<4, 32, 2, 4, false, false, true>, cudaFuncCachePreferShared);
                            conv_img_acts_manycolor2<4, 32, 2, 4, false, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
													  numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else {
                            cudaFuncSetCacheConfig(conv_img_acts_manycolor2<4, 32, 2, 2, false, false, true>, cudaFuncCachePreferShared);
                            conv_img_acts_manycolor2<4, 32, 2, 2, false, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
													  numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        }
                    }
                } else {
                    if (checkCaseBounds) {
                        if (numFilterColors % 16 == 0) {
                            cudaFuncSetCacheConfig(conv_img_acts_manycolor2<4, 32, 1, 4, false, true, true>, cudaFuncCachePreferShared);
                            conv_img_acts_manycolor2<4, 32, 1, 4, false, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
													 numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else {
                            cudaFuncSetCacheConfig(conv_img_acts_manycolor2<4, 32, 1, 2, false, true, true>, cudaFuncCachePreferShared);
                            conv_img_acts_manycolor2<4, 32, 1, 2, false, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
													 numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        }
                    } else {
                        if (numFilterColors % 16 == 0) {
                            cudaFuncSetCacheConfig(conv_img_acts_manycolor2<4, 32, 1, 4, false, false, true>, cudaFuncCachePreferShared);
                            conv_img_acts_manycolor2<4, 32, 1, 4, false, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
													  numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else {
                            cudaFuncSetCacheConfig(conv_img_acts_manycolor2<4, 32, 1, 2, false, false, true>, cudaFuncCachePreferShared);
                            conv_img_acts_manycolor2<4, 32, 1, 2, false, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
													  numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        }
                    }
                }
            } else if (numFilterColors > 3) {
                if (imgsPerThread == 8) {
                    if (checkCaseBounds) {
                        if (colorsPerThread == 4) {
                            cudaFuncSetCacheConfig(img_acts_mediumcolor2<8, 4, false, true, true>, cudaFuncCachePreferShared);
                            img_acts_mediumcolor2<8, 4, false, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											       numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else {
                            cudaFuncSetCacheConfig(img_acts_mediumcolor2<8, 2, false, true, true>, cudaFuncCachePreferShared);
                            img_acts_mediumcolor2<8, 2, false, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											       numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        }
                    } else {
                        if (colorsPerThread == 4) {
                            cudaFuncSetCacheConfig(img_acts_mediumcolor2<8, 4, false, false, true>, cudaFuncCachePreferShared);
                            img_acts_mediumcolor2<8, 4, false, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
												numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else {
                            cudaFuncSetCacheConfig(img_acts_mediumcolor2<8, 2, false, false, true>, cudaFuncCachePreferShared);
                            img_acts_mediumcolor2<8, 2, false, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
												numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        }
                    }
                } else if (imgsPerThread == 4) {
                    if (checkCaseBounds) {
                        if (colorsPerThread == 4) {
                            cudaFuncSetCacheConfig(img_acts_mediumcolor2<4, 4, false, true, true>, cudaFuncCachePreferShared);
                            img_acts_mediumcolor2<4, 4, false, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											       numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else {
                            cudaFuncSetCacheConfig(img_acts_mediumcolor2<4, 2, false, true, true>, cudaFuncCachePreferShared);
                            img_acts_mediumcolor2<4, 2, false, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											       numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        }
                    } else {
                        if (colorsPerThread == 4) {
                            cudaFuncSetCacheConfig(img_acts_mediumcolor2<4, 4, false, false, true>, cudaFuncCachePreferShared);
                            img_acts_mediumcolor2<4, 4, false, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
												numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else {
                            cudaFuncSetCacheConfig(img_acts_mediumcolor2<4, 2, false, false, true>, cudaFuncCachePreferShared);
                            img_acts_mediumcolor2<4, 2, false, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
												numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        }
                    }
                } else {
                    if (checkCaseBounds) {
                        if (colorsPerThread == 4) {
                            cudaFuncSetCacheConfig(img_acts_mediumcolor2<2, 4, false, true, true>, cudaFuncCachePreferShared);
                            img_acts_mediumcolor2<2, 4, false, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											       numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else {
                            cudaFuncSetCacheConfig(img_acts_mediumcolor2<2, 2, false, true, true>, cudaFuncCachePreferShared);
                            img_acts_mediumcolor2<2, 2, false, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											       numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        }
                    } else {
                        if (colorsPerThread == 4) {
                            cudaFuncSetCacheConfig(img_acts_mediumcolor2<2, 4, false, false, true>, cudaFuncCachePreferShared);
                            img_acts_mediumcolor2<2, 4, false, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
												numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else {
                            cudaFuncSetCacheConfig(img_acts_mediumcolor2<2, 2, false, false, true>, cudaFuncCachePreferShared);
                            img_acts_mediumcolor2<2, 2, false, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
												numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        }
                    }
                }
            } else {
                if (imgsPerThread == 8) {
                    if (checkCaseBounds) {
                        if (numFilterColors == 1) {
                            cudaFuncSetCacheConfig(img_acts_color2<8, 1, false, true, true>, cudaFuncCachePreferShared);
                            img_acts_color2<8, 1, false, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											 numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else if (numFilterColors == 2) {
                            cudaFuncSetCacheConfig(img_acts_color2<8, 2, false, true, true>, cudaFuncCachePreferShared);
                            img_acts_color2<8, 2, false, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											 numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else if (numFilterColors == 3) {
                            cudaFuncSetCacheConfig(img_acts_color2<8, 3, false, true, true>, cudaFuncCachePreferShared);
                            img_acts_color2<8, 3, false, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											 numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        }
                    } else {
                        if (numFilterColors == 1) {
                            cudaFuncSetCacheConfig(img_acts_color2<8, 1, false, false, true>, cudaFuncCachePreferShared);
                            img_acts_color2<8, 1, false, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											  numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else if (numFilterColors == 2) {
                            cudaFuncSetCacheConfig(img_acts_color2<8, 2, false, false, true>, cudaFuncCachePreferShared);
                            img_acts_color2<8, 2, false, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											  numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else if (numFilterColors == 3) {
                            cudaFuncSetCacheConfig(img_acts_color2<8, 3, false, false, true>, cudaFuncCachePreferShared);
                            img_acts_color2<8, 3, false, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											  numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        }
                    }
                } else if (imgsPerThread == 4) {
                    if (checkCaseBounds) {
                        if (numFilterColors == 1) {
                            cudaFuncSetCacheConfig(img_acts_color2<4, 1, false, true, true>, cudaFuncCachePreferShared);
                            img_acts_color2<4, 1, false, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											 numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else if (numFilterColors == 2) {
                            cudaFuncSetCacheConfig(img_acts_color2<4, 2, false, true, true>, cudaFuncCachePreferShared);
                            img_acts_color2<4, 2, false, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											 numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else if (numFilterColors == 3) {
                            cudaFuncSetCacheConfig(img_acts_color2<4, 3, false, true, true>, cudaFuncCachePreferShared);
                            img_acts_color2<4, 3, false, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											 numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        }
                    } else {
                        if (numFilterColors == 1) {
                            cudaFuncSetCacheConfig(img_acts_color2<4, 1, false, false, true>, cudaFuncCachePreferShared);
                            img_acts_color2<4, 1, false, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											  numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else if (numFilterColors == 2) {
                            cudaFuncSetCacheConfig(img_acts_color2<4, 2, false, false, true>, cudaFuncCachePreferShared);
                            img_acts_color2<4, 2, false, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											  numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else if (numFilterColors == 3) {
                            cudaFuncSetCacheConfig(img_acts_color2<4, 3, false, false, true>, cudaFuncCachePreferShared);
                            img_acts_color2<4, 3, false, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											  numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        }
                    }
                } else {
                    if (checkCaseBounds) {
                        if (numFilterColors == 1) {
                            cudaFuncSetCacheConfig(img_acts_color2<2, 1, false, true, true>, cudaFuncCachePreferShared);
                            img_acts_color2<2, 1, false, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											 numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else if (numFilterColors == 2) {
                            cudaFuncSetCacheConfig(img_acts_color2<2, 2, false, true, true>, cudaFuncCachePreferShared);
                            img_acts_color2<2, 2, false, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											 numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else if (numFilterColors == 3) {
                            cudaFuncSetCacheConfig(img_acts_color2<2, 3, false, true, true>, cudaFuncCachePreferShared);
                            img_acts_color2<2, 3, false, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											 numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        }
                    } else {
                        if (numFilterColors == 1) {
                            cudaFuncSetCacheConfig(img_acts_color2<2, 1, false, false, true>, cudaFuncCachePreferShared);
                            img_acts_color2<2, 1, false, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											  numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else if (numFilterColors == 2) {
                            cudaFuncSetCacheConfig(img_acts_color2<2, 2, false, false, true>, cudaFuncCachePreferShared);
                            img_acts_color2<2, 2, false, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											  numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else if (numFilterColors == 3) {
                            cudaFuncSetCacheConfig(img_acts_color2<2, 3, false, false, true>, cudaFuncCachePreferShared);
                            img_acts_color2<2, 3, false, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											  numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        }
                    }
                }
            }
        } else { // do scale
            if (numFilterColors % 8 == 0) {
                if (imgsPerThread == 4) {
                    if (checkCaseBounds) {
                        if (numFilterColors % 16 == 0) {
                            cudaFuncSetCacheConfig(conv_img_acts_manycolor2<4, 32, 4, 4, true, true, true>, cudaFuncCachePreferShared);
                            conv_img_acts_manycolor2<4, 32, 4, 4, true, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
													numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else {
                            cudaFuncSetCacheConfig(conv_img_acts_manycolor2<4, 32, 4, 2, true, true, true>, cudaFuncCachePreferShared);
                            conv_img_acts_manycolor2<4, 32, 4, 2, true, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
													numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        }
                    } else {
                        if (numFilterColors % 16 == 0) {
                            cudaFuncSetCacheConfig(conv_img_acts_manycolor2<4, 32, 4, 4, true, false, true>, cudaFuncCachePreferShared);
                            conv_img_acts_manycolor2<4, 32, 4, 4, true, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
													 numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else {
                            cudaFuncSetCacheConfig(conv_img_acts_manycolor2<4, 32, 4, 2, true, false, true>, cudaFuncCachePreferShared);
                            conv_img_acts_manycolor2<4, 32, 4, 2, true, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
													 numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        }
                    }
                } else if (imgsPerThread == 2) {
                    if (checkCaseBounds) {
                        if (numFilterColors % 16 == 0) {
                            cudaFuncSetCacheConfig(conv_img_acts_manycolor2<4, 32, 2, 4, true, true, true>, cudaFuncCachePreferShared);
                            conv_img_acts_manycolor2<4, 32, 2, 4, true, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
													numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else {
                            cudaFuncSetCacheConfig(conv_img_acts_manycolor2<4, 32, 2, 2, true, true, true>, cudaFuncCachePreferShared);
                            conv_img_acts_manycolor2<4, 32, 2, 2, true, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
													numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        }
                    } else {
                        if (numFilterColors % 16 == 0) {
                            cudaFuncSetCacheConfig(conv_img_acts_manycolor2<4, 32, 2, 4, true, false, true>, cudaFuncCachePreferShared);
                            conv_img_acts_manycolor2<4, 32, 2, 4, true, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
													 numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else {
                            cudaFuncSetCacheConfig(conv_img_acts_manycolor2<4, 32, 2, 2, true, false, true>, cudaFuncCachePreferShared);
                            conv_img_acts_manycolor2<4, 32, 2, 2, true, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
													 numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        }
                    }
                } else {
                    if (checkCaseBounds) {
                        if (numFilterColors % 16 == 0) {
                            cudaFuncSetCacheConfig(conv_img_acts_manycolor2<4, 32, 1, 4, true, true, true>, cudaFuncCachePreferShared);
                            conv_img_acts_manycolor2<4, 32, 1, 4, true, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
													numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else {
                            cudaFuncSetCacheConfig(conv_img_acts_manycolor2<4, 32, 1, 2, true, true, true>, cudaFuncCachePreferShared);
                            conv_img_acts_manycolor2<4, 32, 1, 2, true, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
													numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        }
                    } else {
                        if (numFilterColors % 16 == 0) {
                            cudaFuncSetCacheConfig(conv_img_acts_manycolor2<4, 32, 1, 4, true, false, true>, cudaFuncCachePreferShared);
                            conv_img_acts_manycolor2<4, 32, 1, 4, true, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
													 numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else {
                            cudaFuncSetCacheConfig(conv_img_acts_manycolor2<4, 32, 1, 2, true, false, true>, cudaFuncCachePreferShared);
                            conv_img_acts_manycolor2<4, 32, 1, 2, true, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
													 numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        }
                    }
                }
            } else if (numFilterColors > 3) {
                if (imgsPerThread == 8) {
                    if  (checkCaseBounds) {
                        if (colorsPerThread == 4) {
                            cudaFuncSetCacheConfig(img_acts_mediumcolor2<8, 4, true, true, true>, cudaFuncCachePreferShared);
                            img_acts_mediumcolor2<8, 4, true, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											      numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else {
                            cudaFuncSetCacheConfig(img_acts_mediumcolor2<8, 2, true, true, true>, cudaFuncCachePreferShared);
                            img_acts_mediumcolor2<8, 2, true, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											      numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        }
                    } else {
                        if (colorsPerThread == 4) {
                            cudaFuncSetCacheConfig(img_acts_mediumcolor2<8, 4, true, false, true>, cudaFuncCachePreferShared);
                            img_acts_mediumcolor2<8, 4, true, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											       numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else {
                            cudaFuncSetCacheConfig(img_acts_mediumcolor2<8, 2, true, false, true>, cudaFuncCachePreferShared);
                            img_acts_mediumcolor2<8, 2, true, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											       numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        }
                    }
                } else if (imgsPerThread == 4) {
                    if  (checkCaseBounds) {
                        if (colorsPerThread == 4) {
                            cudaFuncSetCacheConfig(img_acts_mediumcolor2<4, 4, true, true, true>, cudaFuncCachePreferShared);
                            img_acts_mediumcolor2<4, 4, true, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											      numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else {
                            cudaFuncSetCacheConfig(img_acts_mediumcolor2<4, 2, true, true, true>, cudaFuncCachePreferShared);
                            img_acts_mediumcolor2<4, 2, true, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											      numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        }
                    } else {
                        if (colorsPerThread == 4) {
                            cudaFuncSetCacheConfig(img_acts_mediumcolor2<4, 4, true, false, true>, cudaFuncCachePreferShared);
                            img_acts_mediumcolor2<4, 4, true, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											       numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else {
                            cudaFuncSetCacheConfig(img_acts_mediumcolor2<4, 2, true, false, true>, cudaFuncCachePreferShared);
                            img_acts_mediumcolor2<4, 2, true, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											       numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        }
                    }
                } else {
                    if  (checkCaseBounds) {
                        if (colorsPerThread == 4) {
                            cudaFuncSetCacheConfig(img_acts_mediumcolor2<2, 4, true, true, true>, cudaFuncCachePreferShared);
                            img_acts_mediumcolor2<2, 4, true, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											      numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else {
                            cudaFuncSetCacheConfig(img_acts_mediumcolor2<2, 2, true, true, true>, cudaFuncCachePreferShared);
                            img_acts_mediumcolor2<2, 2, true, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											      numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        }
                    } else {
                        if (colorsPerThread == 4) {
                            cudaFuncSetCacheConfig(img_acts_mediumcolor2<2, 4, true, false, true>, cudaFuncCachePreferShared);
                            img_acts_mediumcolor2<2, 4, true, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											       numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else {
                            cudaFuncSetCacheConfig(img_acts_mediumcolor2<2, 2, true, false, true>, cudaFuncCachePreferShared);
                            img_acts_mediumcolor2<2, 2, true, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											       numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        }
                    }
                }
            } else {
                if (imgsPerThread == 8) {
                    if (checkCaseBounds) {
                        if (numFilterColors == 1) {
                            cudaFuncSetCacheConfig(img_acts_color2<8, 1, true, true, true>, cudaFuncCachePreferShared);
                            img_acts_color2<8, 1, true, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else if (numFilterColors == 2) {
                            cudaFuncSetCacheConfig(img_acts_color2<8, 2, true, true, true>, cudaFuncCachePreferShared);
                            img_acts_color2<8, 2, true, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else if (numFilterColors == 3) {
                            cudaFuncSetCacheConfig(img_acts_color2<8, 3, true, true, true>, cudaFuncCachePreferShared);
                            img_acts_color2<8, 3, true, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        }
                    } else {
                        if (numFilterColors == 1) {
                            cudaFuncSetCacheConfig(img_acts_color2<8, 1, true, false, true>, cudaFuncCachePreferShared);
                            img_acts_color2<8, 1, true, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											 numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else if (numFilterColors == 2) {
                            cudaFuncSetCacheConfig(img_acts_color2<8, 2, true, false, true>, cudaFuncCachePreferShared);
                            img_acts_color2<8, 2, true, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											 numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else if (numFilterColors == 3) {
                            cudaFuncSetCacheConfig(img_acts_color2<8, 3, true, false, true>, cudaFuncCachePreferShared);
                            img_acts_color2<8, 3, true, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											 numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        }
                    }
                } else if (imgsPerThread == 4) {
                    if (checkCaseBounds) {
                        if (numFilterColors == 1) {
                            cudaFuncSetCacheConfig(img_acts_color2<4, 1, true, true, true>, cudaFuncCachePreferShared);
                            img_acts_color2<4, 1, true, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else if (numFilterColors == 2) {
                            cudaFuncSetCacheConfig(img_acts_color2<4, 2, true, true, true>, cudaFuncCachePreferShared);
                            img_acts_color2<4, 2, true, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else if (numFilterColors == 3) {
                            cudaFuncSetCacheConfig(img_acts_color2<4, 3, true, true, true>, cudaFuncCachePreferShared);
                            img_acts_color2<4, 3, true, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        }
                    } else {
                        if (numFilterColors == 1) {
                            cudaFuncSetCacheConfig(img_acts_color2<4, 1, true, false, true>, cudaFuncCachePreferShared);
                            img_acts_color2<4, 1, true, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											 numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else if (numFilterColors == 2) {
                            cudaFuncSetCacheConfig(img_acts_color2<4, 2, true, false, true>, cudaFuncCachePreferShared);
                            img_acts_color2<4, 2, true, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											 numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else if (numFilterColors == 3) {
                            cudaFuncSetCacheConfig(img_acts_color2<4, 3, true, false, true>, cudaFuncCachePreferShared);
                            img_acts_color2<4, 3, true, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											 numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        }
                    }
                } else {
                    if (checkCaseBounds) {
                        if (numFilterColors == 1) {
                            cudaFuncSetCacheConfig(img_acts_color2<2, 1, true, true, true>, cudaFuncCachePreferShared);
                            img_acts_color2<2, 1, true, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else if (numFilterColors == 2) {
                            cudaFuncSetCacheConfig(img_acts_color2<2, 2, true, true, true>, cudaFuncCachePreferShared);
                            img_acts_color2<2, 2, true, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else if (numFilterColors == 3) {
                            cudaFuncSetCacheConfig(img_acts_color2<2, 3, true, true, true>, cudaFuncCachePreferShared);
                            img_acts_color2<2, 3, true, true, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        }
                    } else {
                        if (numFilterColors == 1) {
                            cudaFuncSetCacheConfig(img_acts_color2<2, 1, true, false, true>, cudaFuncCachePreferShared);
                            img_acts_color2<2, 1, true, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											 numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else if (numFilterColors == 2) {
                            cudaFuncSetCacheConfig(img_acts_color2<2, 2, true, false, true>, cudaFuncCachePreferShared);
                            img_acts_color2<2, 2, true, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											 numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        } else if (numFilterColors == 3) {
                            cudaFuncSetCacheConfig(img_acts_color2<2, 3, true, false, true>, cudaFuncCachePreferShared);
                            img_acts_color2<2, 3, true, false, true><<<blocks, threads>>>(hidActs.getDevData(), hidSwitches.getDevData(), filters.getDevData(), targets.getDevData(),
											 numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, mpSizeX, mpStart, mpStride, mpOutputsX, scaleTargets, scaleOutput);
                        }
                    }
                }
            }
        }
    }






    getLastCudaError("imgActs: kernel execution failed");
}

void convImgActs2(NVMatrix& hidActs, NVMatrix& hidSwitches, NVMatrix& filters, NVMatrix& targets,
		  int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups,
		  int mpSizeX, int mpStart, int mpStride,  int mpOutputsX) {
    _imgActs2(hidActs, hidSwitches, filters, targets, imgSizeY, imgSizeX, numModulesY, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX,  mpStart,  mpStride,   mpOutputsX, 0, 1, true);
}

void convImgActs2(NVMatrix& hidActs, NVMatrix& hidSwitches, NVMatrix& filters, NVMatrix& targets,
		 int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups,
		  int mpSizeX, int mpStart, int mpStride,  int mpOutputsX,
		 float scaleTargets, float scaleOutput) {
    _imgActs2(hidActs, hidSwitches, filters, targets, imgSizeY, imgSizeX, numModulesY, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX,  mpStart,  mpStride,   mpOutputsX,scaleTargets, scaleOutput, true);
}