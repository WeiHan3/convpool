
#include <helper_cuda.h>
#include <nvmatrix.cuh>
#include <cudaconv2.cuh>


/* Atomatical operation, compare two <value, idx> tuples based on the value
 * reference: stackoverflow.com/questions/17411493
 *
 */
typedef union {
    float floats[2];
    int ints[2];
    unsigned long long int ulong;
} ValIdx;

__device__ unsigned long long int my_atomicMax(unsigned long long int* address, float val1, int val2) {
    ValIdx loc, loctest;
    loc.floats[0] = val1;
    loc.ints[1] = val2;
    loctest.ulong = *address;
    while (loctest.floats[0] < val1) 
        loctest.ulong = atomicCAS(address, loctest.ulong, loc.ulong);
    return loctest.ulong;
} 

/* sepearte two interlaced arrays
 * each block handle length/ num blocks pixels
 */
const int DEINTERLACE_THREADS = 256;

__global__ void kDeinterlace(const float *a, const int length, float* const dest1, float* const _dest2) {
     const uint idxX = blockIdx.x * DEINTERLACE_THREADS + threadIdx.x;
     const ValIdx * aa = (const ValIdx *) a;
     int *dest2 = (int *) _dest2;
     if (idxX < length) {
	 dest1[idxX] = aa[idxX].floats[0];
	 dest2[idxX] = aa[idxX].ints[1];
     }     
 }


void _deinterlace(NVMatrix &interlaced, NVMatrix &output1, NVMatrix &output2) {
    assert(output1.getNumElements() == output2.getNumElements());
    assert(output1.getNumElements()*2 == interlaced.getNumElements());
    //    printf("interlaced matrix %f, %f\n", interlaced.max(), interlaced.min()); 
    int numBlocks = DIVUP(output1.getNumElements(), DEINTERLACE_THREADS);
    kDeinterlace<<<numBlocks, DEINTERLACE_THREADS>>>(interlaced.getDevData(), output1.getNumElements(), output1.getDevData(), output2.getDevData());
    getLastCudaError("deinterlace: deinterlace kernel execution failed");

    //    printf("output1 matrix %f, %f\n", output1.max(), output1.min()); 
    //    printf("output2 matrix %f, %f\n", output2.max(), output2.min()); 
}
/*
 * Block size B_YxB_X. Each block applies B_Y * filtersPerThread filters to B_X * imgsPerThread images.
 * threadIdx.x determines image
 * threadIdx.y determines filter
 *
 * blockIdx.x determines image batch of B_X * imgsPerThread
 * blockIdx.y determines filter batch of module and B_Y * filtersPerThread
 *
 * images:      (numColors, imgSizeY, imgSizeX, numImages) with stride given
 * filters:     (numColors, filterPixels, numFilters) if conv
 *              (numModules, numColors, filterPixels, numFilters) otherwise
 *
 * (in filterActs, doesnot really writeout) targets:     (numFilters, numModulesY, numModulesX, numImages)
 * maxPooltargets: (numFilters, mpOutputY, mpOutputX, numImages)
 *
 * B_Y one of 4, 8, 16
 * B_X one of 16, 32
 * imgsPerThread one of 1, 2, 4
 * filtersPerThread one of 1, 2, 4, 8
 *
 * Number of filters per module should be divisible by B_Y * filtersPerThread
 * checkImgBounds indicates whether number of images is divisible by B_X * imgsPerThread
 *
 * The imgSize here is the size of the actual image without the padding.
 *
 */
template <int B_Y, int B_X, int imgsPerThread, int filtersPerThread, int numColors,
          bool scale, bool checkImgBounds>
__global__ void filterActs_YxX_maxPool_color(float* images, float* filters, unsigned long long int* maxPoolTargets,
                                   const int numImages, const int numFilters,
                                   const int imgSizeY, const int imgSizeX, const int filterSize, const int paddingStart,
                                   const int moduleStride,
                                   const int numModulesY, const int numModulesX, const int imgStride,
                                             const int mpSizeX, const int mpStart, const int mpStride, const int mpOutputsX,
                                             //                                   const float scaleTargets, const float scaleOutputs,
                                   const bool conv) {

    __shared__ float shFilters[B_Y*numColors][B_Y * filtersPerThread]; // pre-load B_Y pixels from B_Y*filtersPerThread filters
    __shared__ float shImages[B_Y*numColors][B_X * imgsPerThread]; // pre-load B_Y pixels from B_X*imgsPerThread images
    const int imgPixels = imgSizeY * imgSizeX;
    const int filterPixels = filterSize * filterSize;

    const int blocksPerModule = numFilters / (B_Y*filtersPerThread);
    const int moduleIdx = blockIdx.y / blocksPerModule;
    const int blockFilterIdx = blockIdx.y % blocksPerModule;

    const int tidx = threadIdx.y * B_X + threadIdx.x;

    const int imgLoadModPosY = (moduleIdx / numModulesX) * moduleStride;
    const int imgLoadModPosX = (moduleIdx % numModulesX) * moduleStride;

    const int shFilterLoadY = tidx / (B_Y * filtersPerThread);
    const int shFilterLoadX = tidx % (B_Y * filtersPerThread);
    const int myImgIdx = blockIdx.x * B_X * imgsPerThread + threadIdx.x;
    images += myImgIdx;
    filters += filtersPerThread * B_Y * blockFilterIdx
             + shFilterLoadY * numFilters + shFilterLoadX;
    if (!conv) {
        filters += moduleIdx * numColors * filterPixels * numFilters;
    }

    // moduleIdx should be modposY * numModulesY + modeposX 
    //targets += moduleIdx * numImages
    //        + (blockFilterIdx * B_Y * filtersPerThread + threadIdx.y) * numImages * numModulesY * numModulesX
    //        + myImgIdx;

    
    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for(int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for(int g = 0; g < imgsPerThread; g++) {
            prod[f][g] = 0;
        }
    }

    for (int p = 0; p < filterPixels; p += B_Y) {
        /*
         * Load B_Y pixels from B_Y*filtersPerThread filters
         */
        if (shFilterLoadY < B_Y) {
            #pragma unroll
            for (int p2 = 0; p2 < B_Y; p2 += B_X/filtersPerThread) {
                if (p + p2 + shFilterLoadY < filterPixels) {
                    #pragma unroll
                    for (int c = 0; c < numColors; c++) {
                        shFilters[shFilterLoadY + p2 + c * B_Y][shFilterLoadX] = filters[(c * filterPixels + p + p2) * numFilters];
                    }
                } else {
                    #pragma unroll
                    for (int c = 0; c < numColors; c++) {
                        shFilters[shFilterLoadY + p2 + c * B_Y][shFilterLoadX] = 0;
                    }
                }
            }
        }

        /*
         * Load B_Y pixels from B_X*imgsPerThread images
         */
        const int pixIdx = p + threadIdx.y;
        if (pixIdx < filterPixels) {
            const int x = paddingStart + imgLoadModPosX + pixIdx % filterSize;
            const int y = paddingStart + imgLoadModPosY + pixIdx / filterSize;
            if (y >= 0 && y< imgSizeY && x >= 0 && x < imgSizeX) {
                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
                        #pragma unroll
                        for (int c = 0; c < numColors; c++) {
                            shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = images[imgStride * (c * imgPixels + y * imgSizeX + x) + i * B_X];
                        }
                    } else {
                        #pragma unroll
                        for (int c = 0; c < numColors; c++) {
                            shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = 0;
                        }
                    }
                }
            } else { // Padding
                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    #pragma unroll
                    for (int c = 0; c < numColors; c++) {
                        shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = 0;
                    }
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < B_Y*numColors; i++) {
            #pragma unroll
            for(int f = 0; f < filtersPerThread; f++) {
                #pragma unroll
                for(int g = 0; g < imgsPerThread; g++) {
                    prod[f][g] += shImages[i][g * B_X + threadIdx.x] * shFilters[i][threadIdx.y + f * B_Y];
                }
            }

        }
        __syncthreads();
    }

    maxPoolTargets += (blockFilterIdx * B_Y * filtersPerThread + threadIdx.y) * numImages * mpOutputsX * mpOutputsX + myImgIdx;
    const int convTargetY = moduleIdx / numModulesX;
    const int convTargetX = moduleIdx % numModulesX;
    
    /*   if (scale) {
        #pragma unroll
        for (int g = 0; g < imgsPerThread; g++) {
            if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    targets[g * B_X + f * B_Y * numImages * numModulesY * numModulesX] = scaleTargets * targets[g * B_X + f * B_Y * numImages * numModulesY * numModulesX] + scaleOutputs * prod[f][g];
                }
            }
        }
        } else {*/
    const int loopStartY = convTargetY - mpStart < mpSizeX ? 0 : 1 + (convTargetY - mpStart - mpSizeX) / mpStride;
    const int loopStartX = convTargetX - mpStart < mpSizeX ? 0 : 1 + (convTargetX - mpStart - mpSizeX) / mpStride;

    const int loopEndY = MIN(mpOutputsX-1, (convTargetY - mpStart) / mpStride);
    const int loopEndX = MIN(mpOutputsX-1, (convTargetX - mpStart) / mpStride);
    for (int y = loopStartY; y <= loopEndY; y++) {
        for (int x = loopStartX; x <= loopEndX; x++) {
            const int mpOutIdx = y * mpOutputsX + x;
            #pragma unroll
            for (int g = 0; g < imgsPerThread; g++) {
                if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
            //targets[g * B_X + f * B_Y * numImages * numModulesY * numModulesX] = scaleOutputs * prod[f][g];
                // moduleIdx should be modposY * numModulesY + modeposX 
                // this filter is f*B_Y + filter_base
                        //maxPoolTargets[mpOutIdx*numImages + g * B_X + f * B_Y * numImages * mpOutputsX * mpOutputsX];
			//atomicMax((float *)&maxPoolTargets[mpOutIdx * numImages + g * B_X + f * B_Y * numImages * mpOutputsX * mpOutputsX], prod[f][g]);
			//maxPoolTargets[mpOutIdx * numImages + g * B_X + f * B_Y * numImages * mpOutputsX * mpOutputsX] = prod[f][g];
                        my_atomicMax(&maxPoolTargets[mpOutIdx * numImages + g * B_X + f * B_Y * numImages * mpOutputsX * mpOutputsX], prod[f][g], moduleIdx);
                    }
                }
            }
            
        }
    }

}


/*
 * Block size B_YxB_X. Each block applies B_Y * filtersPerThread filters to B_X * imgsPerThread images.
 * threadIdx.x determines image
 * threadIdx.y determines filter
 *
 * blockIdx.x determines image batch of B_X * imgsPerThread
 * blockIdx.y determines filter batch of B_Y * filtersPerThread
 *
 * images:      (numImgColors, imgSizeY, imgSizeX, numImages) with stride given
 * filters:     (numFilterColors, filterPixels, numFilters) if conv
 *              (numModules, numFilterColors, filterPixels, numFilters) otherwise
 *
 * targets:     (numFilters, numModulesY, numModulesX, numImages)
 *
 * B_Y one of 4, 8, 16
 * B_X one of 16, 32
 * imgsPerThread one of 1, 2, 4
 * filtersPerThread one of 1, 2, 4, 8
 * colorCache: how many colors to put into shmem
 *
 * numFilters should be divisible by B_Y * filtersPerThread
 * numImages be divisible by B_X * imgsPerThread
 * numFilterColors should be divisible by colorCache.
 * numImgColors must be even.
 * numFilters must be divisible by numGroups.
 *
 * The imgSize here is the size of the actual image without the padding.
 *
 */
template <int B_Y, int B_X, int imgsPerThread, int filtersPerThread, int colorCache,
          bool scale, bool checkImgBounds>
__global__ void filterActs_YxX_maxPool_sparse(float* images, float* filters, unsigned long long int * maxPoolTargets,
                                       const int numImages, const int numFilters,
                                       const int imgSizeY, const int imgSizeX, const int filterSize, const int paddingStart,
                                       const int moduleStride,
                                       const int numModulesY, const int numModulesX, const int imgStride, const int numImgColors,
                                       const int numGroups, 
                                              const int mpSizeX, const int mpStart, const int mpStride, const int mpOutputsX,
                                              //                                       const float scaleTargets, const float scaleOutputs,
                                       const bool conv) {
    __shared__ float shFilters[B_Y*colorCache][B_Y * filtersPerThread]; // pre-load B_Y pixels from B_Y*filtersPerThread filters
    __shared__ float shImages[B_Y*colorCache][B_X * imgsPerThread]; // pre-load B_Y pixels from B_X*imgsPerThread images
    const int imgPixels = imgSizeY * imgSizeX;
    const int filterPixels = filterSize * filterSize;
    const int numFilterColors = numImgColors / numGroups;
    const int blocksPerModule = numFilters / (B_Y*filtersPerThread);
    const int moduleIdx = blockIdx.y / blocksPerModule;
    const int blockFilterIdx = filtersPerThread * B_Y * (blockIdx.y % blocksPerModule);
    const int numFiltersPerGroup = numFilters / numGroups;
    const int blockGroupIdx = blockFilterIdx / numFiltersPerGroup;

    //    const int numModules = numModulesX * numModulesY;
    const int blockColorIdx = numFilterColors * blockGroupIdx;

    const int tidx = threadIdx.y * B_X + threadIdx.x;

    const int imgLoadModPosY = paddingStart + (moduleIdx / numModulesX) * moduleStride;
    const int imgLoadModPosX = paddingStart + (moduleIdx % numModulesX) * moduleStride;

    const int shFilterLoadY = tidx / (B_Y * filtersPerThread);
    const int shFilterLoadX = tidx % (B_Y * filtersPerThread);
    const int myImgIdx = blockIdx.x * B_X * imgsPerThread + threadIdx.x;

    images += blockColorIdx * imgPixels * imgStride + myImgIdx;
    filters +=blockFilterIdx
            + shFilterLoadY * numFilters + shFilterLoadX;
    if (!conv) {
        filters += moduleIdx * numFilterColors * filterPixels * numFilters;
    }

    //    targets += moduleIdx * numImages
    //        + (blockFilterIdx + threadIdx.y) * numImages * numModules
    //       + myImgIdx;

    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for(int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for(int g = 0; g < imgsPerThread; g++) {
            prod[f][g] = 0;
        }
    }
//    __shared__ int imgPos[]
    for (int oc = 0; oc < numFilterColors; oc += colorCache) { // oc stands for outer color (loop)
        for (int p = 0; p < filterPixels; p += B_Y) {
            /*
             * Load B_Y pixels from B_Y*filtersPerThread filters
             */
            if (shFilterLoadY < B_Y) {
                #pragma unroll
                for (int p2 = 0; p2 < B_Y; p2 += B_X/filtersPerThread) {
                    if (p + p2 + shFilterLoadY < filterPixels) {
                        #pragma unroll
                        for (int c = 0; c < colorCache; c++) {
                            shFilters[shFilterLoadY + p2 + c * B_Y][shFilterLoadX] = filters[((oc+c) * filterPixels + p + p2) * numFilters];
                        }
                    } else {
                        #pragma unroll
                        for (int c = 0; c < colorCache; c++) {
                            shFilters[shFilterLoadY + p2 + c * B_Y][shFilterLoadX] = 0;
                        }
                    }
                }
            }

            /*
             * Load B_Y pixels from B_X*imgsPerThread images
             */
            const int pixIdx = p + threadIdx.y;
            if (pixIdx < filterPixels) {
                const int x = imgLoadModPosX + pixIdx % filterSize;
                const int y = imgLoadModPosY + pixIdx / filterSize;
                if (y >= 0 && y < imgSizeY && x >= 0 && x < imgSizeX) {
                    float* m = &images[imgStride * (oc * imgPixels + y * imgSizeX + x)];
                    #pragma unroll
                    for (int i = 0; i < imgsPerThread; i++) {
                        if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
                            #pragma unroll
                            for (int c = 0; c < colorCache; c++) {
                                shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = m[c * imgStride * imgPixels + i * B_X];
                            }
                        } else {
                            #pragma unroll
                            for (int c = 0; c < colorCache; c++) {
                                shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = 0;
                            }
                        }
                    }
                } else { // Padding
                    #pragma unroll
                    for (int i = 0; i < imgsPerThread; i++) {
                        #pragma unroll
                        for (int c = 0; c < colorCache; c++) {
                            shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = 0;
                        }
                    }
                }
            }
            __syncthreads();
            #pragma unroll
            for (int i = 0; i < B_Y*colorCache; i++) {
                #pragma unroll
                for(int f = 0; f < filtersPerThread; f++) {
                    #pragma unroll
                    for(int g = 0; g < imgsPerThread; g++) {
                        prod[f][g] += shImages[i][g * B_X + threadIdx.x] * shFilters[i][threadIdx.y + f * B_Y];
                    }
                }

            }
            __syncthreads();
        }
    }

    maxPoolTargets += (blockFilterIdx + threadIdx.y) * numImages * mpOutputsX * mpOutputsX + myImgIdx;
    const int convTargetY = moduleIdx / numModulesX;
    const int convTargetX = moduleIdx % numModulesX;

    /*    if (scale) {
        #pragma unroll
        for (int g = 0; g < imgsPerThread; g++) {
            if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    targets[g * B_X + f * B_Y * numImages * numModules] = scaleTargets * targets[g * B_X + f * B_Y * numImages * numModules] + scaleOutputs * prod[f][g];
                }
            }
        }
    } else {
        #pragma unroll
        for (int g = 0; g < imgsPerThread; g++) {
            if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    targets[g * B_X + f * B_Y * numImages * numModules] = scaleOutputs * prod[f][g];
                }
            }
        }
        }*/

    const int loopStartY = MAX(0, DIVUP(convTargetY - mpStart - (mpSizeX-1),  mpStride));
    const int loopStartX = MAX(0, DIVUP(convTargetX - mpStart - (mpSizeX-1),  mpStride));
    const int loopEndY = MIN(mpOutputsX-1, (convTargetY - mpStart) / mpStride);
    const int loopEndX = MIN(mpOutputsX-1, (convTargetX - mpStart) / mpStride);



    for (int y = loopStartY; y <= loopEndY; y++) {
        for (int x = loopStartX; x <= loopEndX; x++) {
            const int mpOutIdx = y * mpOutputsX + x;
            #pragma unroll
            for (int g = 0; g < imgsPerThread; g++) {
                if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
            //targets[g * B_X + f * B_Y * numImages * numModulesY * numModulesX] = scaleOutputs * prod[f][g];
                // moduleIdx should be modposY * numModulesY + modeposX 
                // this filter is f*B_Y + filter_base
                        //maxPoolTargets[mpOutIdx*numImages + g * B_X + f * B_Y * numImages * mpOutputsX * mpOutputsX];
			//maxPoolTargets[mpOutIdx * numImages + g * B_X + f * B_Y * numImages * mpOutputsX * mpOutputsX] =  prod[f][g];
			my_atomicMax(&maxPoolTargets[mpOutIdx * numImages + g * B_X + f * B_Y * numImages * mpOutputsX * mpOutputsX], prod[f][g], moduleIdx);
                    }
                }
            }
            
        }
    }
    
}


/*
 * images:      (numImgColors, imgSizeY, imgSizeX, numImages) with stride given
 * filters:     (numFilterColors, filterPixels, numFilters)             if conv
 *              (numModules, numFilterColors, filterPixels, numFilters) otherwise
 *
 * targets:     (numFilters, numModules, numImages)
 * maxPoolTargets: (numFilters, mpOuputsX*mpOutputX, numImage)
 * 
 * Note: all of these convolution routines are optimized for the case when
 * the number of images (i.e. the minibatch size) is a multiple of 128. 
 * Other batch sizes will work, but but I made no attempt whatsoever
 * to make them work fast. 
 */
void _filterActs_maxPool(NVMatrix& images, NVMatrix& filters, NVMatrix& targetsVal, NVMatrix& targetsSwitch,
                   int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                   int numImgColors, int numGroups,
                  //                   float scaleTargets, float scaleOutput, 
                          const int mpSizeX, const int mpStart, const int mpStride, const int mpOutputsX,
                  bool conv) {
    int numFilterColors = numImgColors / numGroups;      
    int numFilters = filters.getNumCols();
    int numModules = numModulesY * numModulesX;
    int numImages = images.getNumCols();
    int imgPixels = images.getNumRows()/numImgColors;
    int imgSizeX = imgPixels / imgSizeY;
    int filterModuleMult = conv ? 1 : numModules;
    
    assert(numGroups > 1 || (numImgColors > 0 && (numImgColors <= 3 || numImgColors % 2 == 0)));
    assert(numGroups == 1 || numFilterColors % 2 == 0);
    assert(numFilters % (16 * numGroups) == 0);
    assert(numImgColors % numGroups == 0);
    assert(images.getNumRows() == imgPixels * numImgColors);
    assert(imgSizeY * imgSizeX == imgPixels);
    int numFiltersPerGroup = numFilters / numGroups;

    int imgStride = images.getStride(); // images does not need to be a contiguous matrix

    int filterPixels = filters.getNumRows() / (filterModuleMult * numFilterColors);
    int filterSize = int(sqrt(filterPixels));
    assert(filterSize * filterSize == filterPixels);
    assert(filters.getNumRows() == filterModuleMult * numFilterColors * filterPixels);

    // These routines don't handle the case when only part of the image is visited in the convolution
    assert(paddingStart <= 0);
    assert(paddingStart + (numModulesX-1)*moduleStride + filterSize >= imgSizeX);
    assert(paddingStart + (numModulesY-1)*moduleStride + filterSize >= imgSizeY);
    assert(moduleStride <= filterSize);
    
    assert(!images.isTrans());
    assert(!filters.isTrans());
    assert(!targetsVal.isTrans());
    assert(!targetsSwitch.isTrans());

    assert(filters.isContiguous());
    assert(targetsVal.isContiguous());
    assert(targetsSwitch.isContiguous());

    int imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
    dim3 blocks = numFiltersPerGroup % 32 == 0 ? dim3(DIVUP(numImages, 32 * imgsPerThread), (numModules * numFilters) / (4 * 8))
                                               : dim3(DIVUP(numImages, 32 * imgsPerThread), (numModules * numFilters) / (4 * 4));
    dim3 threads(32, 4);
    bool checkImgBounds = numImages % (32*imgsPerThread) != 0;
    
    
    int outputs = mpOutputsX * mpOutputsX;
    
    NVMatrix& targets = *(new NVMatrix()); // to hold the validx pairs
    targets.resize(numFilters*outputs*2, numImages);

    // maxpooler.getBaseValue() is -2e38
    targets.apply(NVMatrixOps::WeightedAddScalar(0,-2e38)); 

    //    printf("interlaced before convmax: %f, %f\n", targets.max(), targets.min());

    targetsVal.resize(numFilters*outputs, numImages);
    targetsSwitch.resize(numFilters*outputs, numImages);
    
    /*
    if (scaleTargets == 0) {
        targets.resize(numFilters * numModules, numImages);
    } else {
        assert(targets.getNumRows() == numFilters * numModules);
        assert(targets.getNumCols() == numImages);
    }
    */

    unsigned long long int  * tempData = (unsigned long long int *) targets.getDevData();
    if (imgsPerThread == 4) {
        if (numImgColors <= 3) {
            assert(numGroups == 1); // It has to be based on above definitions, but just to be sure.
            //if (scaleTargets == 0) { // don't scale
            if (numImgColors == 1) {
                if (checkImgBounds) {
                    if (numFilters % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_maxPool_color< 4, 32, 4, 8, 1, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_maxPool_color < 4, 32, 4, 8, 1, false, true > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, mpSizeX, mpStart, mpStride, mpOutputsX, conv);
                    } else {
                        cudaFuncSetCacheConfig(filterActs_YxX_maxPool_color< 4, 32, 4, 4, 1, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_maxPool_color < 4, 32, 4, 4, 1, false, true > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                   numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, mpSizeX, mpStart, mpStride, mpOutputsX, conv);

                    }
                } else {
                    if (numFilters % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_maxPool_color< 4, 32, 4, 8, 1, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_maxPool_color < 4, 32, 4, 8, 1, false, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, mpSizeX, mpStart, mpStride, mpOutputsX, conv);

                    } else {
                        cudaFuncSetCacheConfig(filterActs_YxX_maxPool_color< 4, 32, 4, 4, 1, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_maxPool_color < 4, 32, 4, 4, 1, false, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, mpSizeX, mpStart, mpStride, mpOutputsX, conv);
                        
                    }
                }
            } else if (numImgColors == 2) {
                if (checkImgBounds) {
                    if (numFilters % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_maxPool_color< 4, 32, 4, 8, 2, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_maxPool_color < 4, 32, 4, 8, 2, false, true > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                   numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, mpSizeX, mpStart, mpStride, mpOutputsX, conv);
                        
                    } else {
                        cudaFuncSetCacheConfig(filterActs_YxX_maxPool_color< 4, 32, 4, 4, 2, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_maxPool_color < 4, 32, 4, 4, 2, false, true > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                   numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, mpSizeX, mpStart, mpStride, mpOutputsX, conv);

                    }
                } else {
                    if (numFilters % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_maxPool_color< 4, 32, 4, 8, 2, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_maxPool_color < 4, 32, 4, 8, 2, false, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, mpSizeX, mpStart, mpStride, mpOutputsX, conv);

                    } else {
                        cudaFuncSetCacheConfig(filterActs_YxX_maxPool_color< 4, 32, 4, 4, 2, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_maxPool_color < 4, 32, 4, 4, 2, false, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, mpSizeX, mpStart, mpStride, mpOutputsX, conv);
                        
                    }
                }
            }  else if (numImgColors == 3) {
                if (checkImgBounds) {
                    if (numFilters % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_maxPool_color< 4, 32, 4, 8, 3, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_maxPool_color < 4, 32, 4, 8, 3, false, true > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                   numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, mpSizeX, mpStart, mpStride, mpOutputsX, conv);

                    } else {
                        cudaFuncSetCacheConfig(filterActs_YxX_maxPool_color< 4, 32, 4, 4, 3, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_maxPool_color < 4, 32, 4, 4, 3, false, true > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                   numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, mpSizeX, mpStart, mpStride, mpOutputsX, conv);

                    }
                } else {
                    if (numFilters % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_maxPool_color< 4, 32, 4, 8, 3, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_maxPool_color < 4, 32, 4, 8, 3, false, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, mpSizeX, mpStart, mpStride, mpOutputsX, conv);

                    } else {
                        cudaFuncSetCacheConfig(filterActs_YxX_maxPool_color< 4, 32, 4, 4, 3, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_maxPool_color < 4, 32, 4, 4, 3, false, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, mpSizeX, mpStart, mpStride, mpOutputsX, conv);
                        
                    }
                }
            }

        } else {
            
            if (checkImgBounds) {
                if (numFiltersPerGroup % 32 == 0) {
                    cudaFuncSetCacheConfig(filterActs_YxX_maxPool_sparse< 4, 32, 4, 8, 2, false, true >, cudaFuncCachePreferShared);
                    filterActs_YxX_maxPool_sparse< 4, 32, 4, 8, 2, false, true > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
												       numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, conv);
                    
                } else {
                    cudaFuncSetCacheConfig(filterActs_YxX_maxPool_sparse< 4, 32, 4, 4, 2, false, true >, cudaFuncCachePreferShared);
                    filterActs_YxX_maxPool_sparse< 4, 32, 4, 4, 2, false, true > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, conv);
                    
                }
            } else {
                if (numFiltersPerGroup % 32 == 0) {
                    cudaFuncSetCacheConfig(filterActs_YxX_maxPool_sparse< 4, 32, 4, 8, 2, false, false >, cudaFuncCachePreferShared);
                    filterActs_YxX_maxPool_sparse< 4, 32, 4, 8, 2, false, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                 numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, conv);

                } else {
                    cudaFuncSetCacheConfig(filterActs_YxX_maxPool_sparse< 4, 32, 4, 4, 2, false, false >, cudaFuncCachePreferShared);
                    filterActs_YxX_maxPool_sparse< 4, 32, 4, 4, 2, false, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                 numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, conv);
                    
                }
            }
        }
    } else if (imgsPerThread == 2) {
        if (numImgColors <= 3) {
            assert(numGroups == 1); // It has to be based on above definitions, but just to be sure.
            
            if (numImgColors == 1) {
                if (checkImgBounds) {
                    if (numFilters % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_maxPool_color< 4, 32, 2, 8, 1, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_maxPool_color < 4, 32, 2, 8, 1, false, true > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                   numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, mpSizeX, mpStart, mpStride, mpOutputsX, conv);
                        
                    } else {
                        cudaFuncSetCacheConfig(filterActs_YxX_maxPool_color< 4, 32, 2, 4, 1, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_maxPool_color < 4, 32, 2, 4, 1, false, true > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                   numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, mpSizeX, mpStart, mpStride, mpOutputsX, conv);

                    }
                } else {
                    if (numFilters % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_maxPool_color< 4, 32, 2, 8, 1, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_maxPool_color < 4, 32, 2, 8, 1, false, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, mpSizeX, mpStart, mpStride, mpOutputsX, conv);
                        
                    } else {
                        cudaFuncSetCacheConfig(filterActs_YxX_maxPool_color< 4, 32, 2, 4, 1, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_maxPool_color < 4, 32, 2, 4, 1, false, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, mpSizeX, mpStart, mpStride, mpOutputsX, conv);

                    }
                }
            } else if (numImgColors == 2) {
                if (checkImgBounds) {
                    if (numFilters % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_maxPool_color< 4, 32, 2, 8, 2, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_maxPool_color < 4, 32, 2, 8, 2, false, true > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                   numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, mpSizeX, mpStart, mpStride, mpOutputsX, conv);

                    } else {
                        cudaFuncSetCacheConfig(filterActs_YxX_maxPool_color< 4, 32, 2, 4, 2, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_maxPool_color < 4, 32, 2, 4, 2, false, true > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                   numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, mpSizeX, mpStart, mpStride, mpOutputsX, conv);

                    }
                } else {
                    if (numFilters % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_maxPool_color< 4, 32, 2, 8, 2, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_maxPool_color < 4, 32, 2, 8, 2, false, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, mpSizeX, mpStart, mpStride, mpOutputsX, conv);

                    } else {
                        cudaFuncSetCacheConfig(filterActs_YxX_maxPool_color< 4, 32, 2, 4, 2, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_maxPool_color < 4, 32, 2, 4, 2, false, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, mpSizeX, mpStart, mpStride, mpOutputsX, conv);

                    }
                }
            }  else if (numImgColors == 3) {
                if (checkImgBounds) {
                    if (numFilters % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_maxPool_color< 4, 32, 2, 8, 3, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_maxPool_color < 4, 32, 2, 8, 3, false, true > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                   numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, mpSizeX, mpStart, mpStride, mpOutputsX, conv);

                    } else {
                        cudaFuncSetCacheConfig(filterActs_YxX_maxPool_color< 4, 32, 2, 4, 3, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_maxPool_color < 4, 32, 2, 4, 3, false, true > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                   numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, mpSizeX, mpStart, mpStride, mpOutputsX, conv);
                        
                    }
                } else {
                    if (numFilters % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_maxPool_color< 4, 32, 2, 8, 3, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_maxPool_color < 4, 32, 2, 8, 3, false, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, mpSizeX, mpStart, mpStride, mpOutputsX, conv);

                    } else {
                        cudaFuncSetCacheConfig(filterActs_YxX_maxPool_color< 4, 32, 2, 4, 3, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_maxPool_color < 4, 32, 2, 4, 3, false, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, mpSizeX, mpStart, mpStride, mpOutputsX, conv);

                    }
                }
	    }
        } else {
	    
	    if (checkImgBounds) {
		if (numFiltersPerGroup % 32 == 0) {
		    cudaFuncSetCacheConfig(filterActs_YxX_maxPool_sparse< 4, 32, 2, 8, 2, false, true >, cudaFuncCachePreferShared);
		    filterActs_YxX_maxPool_sparse< 4, 32, 2, 8, 2, false, true > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
												numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, conv);

		} else {
		    cudaFuncSetCacheConfig(filterActs_YxX_maxPool_sparse< 4, 32, 2, 4, 2, false, true >, cudaFuncCachePreferShared);
		    filterActs_YxX_maxPool_sparse< 4, 32, 2, 4, 2, false, true > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
												numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, conv);

		}
	    } else {
		if (numFiltersPerGroup % 32 == 0) {
		    cudaFuncSetCacheConfig(filterActs_YxX_maxPool_sparse< 4, 32, 2, 8, 2, false, false >, cudaFuncCachePreferShared);
		    filterActs_YxX_maxPool_sparse< 4, 32, 2, 8, 2, false, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
												 numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numGroups, numImgColors, mpSizeX, mpStart, mpStride, mpOutputsX, conv);

		} else {
		    cudaFuncSetCacheConfig(filterActs_YxX_maxPool_sparse< 4, 32, 2, 4, 2, false, false >, cudaFuncCachePreferShared);
		    filterActs_YxX_maxPool_sparse< 4, 32, 2, 4, 2, false, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
												 numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, conv);
		    
		}
	    }
	}    
    } else {
        if (numImgColors <= 3) {
            assert(numGroups == 1); // It has to be based on above definitions, but just to be sure.
	    if (numImgColors == 1) {
		if (checkImgBounds) {
		    if (numFilters % 32 == 0) {
			cudaFuncSetCacheConfig(filterActs_YxX_maxPool_color< 4, 32, 1, 8, 1, false, true >, cudaFuncCachePreferShared);
			filterActs_YxX_maxPool_color < 4, 32, 1, 8, 1, false, true > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, mpSizeX, mpStart, mpStride, mpOutputsX, conv);

		    } else {
			cudaFuncSetCacheConfig(filterActs_YxX_maxPool_color< 4, 32, 1, 4, 1, false, true >, cudaFuncCachePreferShared);
			filterActs_YxX_maxPool_color < 4, 32, 1, 4, 1, false, true > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, mpSizeX, mpStart, mpStride, mpOutputsX, conv);

		    }
		} else {
		    if (numFilters % 32 == 0) {
			cudaFuncSetCacheConfig(filterActs_YxX_maxPool_color< 4, 32, 1, 8, 1, false, false >, cudaFuncCachePreferShared);
			filterActs_YxX_maxPool_color < 4, 32, 1, 8, 1, false, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, mpSizeX, mpStart, mpStride, mpOutputsX, conv);

		    } else {
			cudaFuncSetCacheConfig(filterActs_YxX_maxPool_color< 4, 32, 1, 4, 1, false, false >, cudaFuncCachePreferShared);
			filterActs_YxX_maxPool_color < 4, 32, 1, 4, 1, false, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, mpSizeX, mpStart, mpStride, mpOutputsX, conv);

		    }
		}
	    } else if (numImgColors == 2) {
		if (checkImgBounds) {
		    if (numFilters % 32 == 0) {
			cudaFuncSetCacheConfig(filterActs_YxX_maxPool_color< 4, 32, 1, 8, 2, false, true >, cudaFuncCachePreferShared);
			filterActs_YxX_maxPool_color < 4, 32, 1, 8, 2, false, true > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, mpSizeX, mpStart, mpStride, mpOutputsX, conv);

		    } else {
			cudaFuncSetCacheConfig(filterActs_YxX_maxPool_color< 4, 32, 1, 4, 2, false, true >, cudaFuncCachePreferShared);
			filterActs_YxX_maxPool_color < 4, 32, 1, 4, 2, false, true > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, mpSizeX, mpStart, mpStride, mpOutputsX, conv);

		    }
		} else {
		    if (numFilters % 32 == 0) {
			cudaFuncSetCacheConfig(filterActs_YxX_maxPool_color< 4, 32, 1, 8, 2, false, false >, cudaFuncCachePreferShared);
			filterActs_YxX_maxPool_color < 4, 32, 1, 8, 2, false, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, mpSizeX, mpStart, mpStride, mpOutputsX, conv);

		    } else {
			cudaFuncSetCacheConfig(filterActs_YxX_maxPool_color< 4, 32, 1, 4, 2, false, false >, cudaFuncCachePreferShared);
			filterActs_YxX_maxPool_color < 4, 32, 1, 4, 2, false, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, mpSizeX, mpStart, mpStride, mpOutputsX, conv);

		    }
		}
	    }  else if (numImgColors == 3) {
		if (checkImgBounds) {
		    if (numFilters % 32 == 0) {
			cudaFuncSetCacheConfig(filterActs_YxX_maxPool_color< 4, 32, 1, 8, 3, false, true >, cudaFuncCachePreferShared);
			filterActs_YxX_maxPool_color < 4, 32, 1, 8, 3, false, true > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, mpSizeX, mpStart, mpStride, mpOutputsX, conv);

		    } else {
			cudaFuncSetCacheConfig(filterActs_YxX_maxPool_color< 4, 32, 1, 4, 3, false, true >, cudaFuncCachePreferShared);
			filterActs_YxX_maxPool_color < 4, 32, 1, 4, 3, false, true > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, mpSizeX, mpStart, mpStride, mpOutputsX, conv);

		    }
		} else {
		    if (numFilters % 32 == 0) {
			cudaFuncSetCacheConfig(filterActs_YxX_maxPool_color< 4, 32, 1, 8, 3, false, false >, cudaFuncCachePreferShared);
			filterActs_YxX_maxPool_color < 4, 32, 1, 8, 3, false, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, mpSizeX, mpStart, mpStride, mpOutputsX, conv);

		    } else {
			cudaFuncSetCacheConfig(filterActs_YxX_maxPool_color< 4, 32, 1, 4, 3, false, false >, cudaFuncCachePreferShared);
			filterActs_YxX_maxPool_color < 4, 32, 1, 4, 3, false, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, mpSizeX, mpStart, mpStride, mpOutputsX, conv);

		    }
		}
		
            }
        } else {
	    
	    if (checkImgBounds) {
		if (numFiltersPerGroup % 32 == 0) {
		    cudaFuncSetCacheConfig(filterActs_YxX_maxPool_sparse< 4, 32, 1, 8, 2, false, true >, cudaFuncCachePreferShared);
		    filterActs_YxX_maxPool_sparse< 4, 32, 1, 8, 2, false, true > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, conv);

		} else {
		    cudaFuncSetCacheConfig(filterActs_YxX_maxPool_sparse< 4, 32, 1, 4, 2, false, true >, cudaFuncCachePreferShared);
		    filterActs_YxX_maxPool_sparse< 4, 32, 1, 4, 2, false, true > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, conv);

		}
	    } else {
		if (numFiltersPerGroup % 32 == 0) {
		    cudaFuncSetCacheConfig(filterActs_YxX_maxPool_sparse< 4, 32, 1, 8, 2, false, false >, cudaFuncCachePreferShared);
		    filterActs_YxX_maxPool_sparse< 4, 32, 1, 8, 2, false, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, conv);

		} else {
		    cudaFuncSetCacheConfig(filterActs_YxX_maxPool_sparse< 4, 32, 1, 4, 2, false, false >, cudaFuncCachePreferShared);
		    filterActs_YxX_maxPool_sparse< 4, 32, 1, 4, 2, false, false > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), tempData,
                                                                                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, conv);

		}
	    }
        }
    }
    getLastCudaError("filterActs: kernel execution failed");
    //    printf("before interlace\n");



    _deinterlace(targets, targetsVal, targetsSwitch);
    //    printf("end interlace\n");
    cudaThreadSynchronize();
    delete &targets;
    //    printf("end delete\n");

}

/*
void convFilterActs(NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
                          int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                          int numImgColors, int numGroups) {
    convFilterActs(images, filters, targets, imgSizeY, numModulesY, numModulesX, paddingStart, moduleStride, numImgColors, numGroups, 0, 1);
}
*/
void convMaxPoolFilterActs(NVMatrix& images, NVMatrix& filters, NVMatrix& targetsVal, NVMatrix &targetsSwitch, 
			    int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
			    int numImgColors, int numGroups,
			    int mpSizeX, int mpStart, int mpStride, int mpOutputsX) {
    _filterActs_maxPool(images, filters, targetsVal, targetsSwitch, imgSizeY, numModulesY, numModulesX, paddingStart, moduleStride, numImgColors, numGroups, mpSizeX, mpStart, mpStride, mpOutputsX, true);
}

/*void localFilterActs(NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
                          int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                          int numImgColors, int numGroups) {
    localFilterActs(images, filters, targets, imgSizeY, numModulesY, numModulesX, paddingStart, moduleStride, numImgColors, numGroups, 0, 1);
}

void localFilterActs(NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
                   int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                   int numImgColors, int numGroups,
                   float scaleTargets, float scaleOutput) {
     _filterActs(images, filters, targets, imgSizeY, numModulesY, numModulesX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, false);
}
*/