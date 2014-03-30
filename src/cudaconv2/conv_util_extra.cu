
#include <iostream>
#include <assert.h>
#include <nvmatrix_kernels.cuh>
#include <nvmatrix.cuh>
#include <conv_util.cuh>

using namespace std;

/*
 * Block size B_YxB_X
 * blockIdx.x determines pixel.x, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines pixel.y, filter idx in batches of B_Y*filtersPerThread
 * 
 * So each block does one output pixel for some number of images/filters.
 * 
 * threadIdx.x determines img idx
 * threadIdx.y determines filter idx
 * 
 * maxGrads:    (numFilters, numOutputs, numImages)
 * maxInds:    (numFilters, numOutputs, numImages)
 * target:      (numFilters, imgPixels, numImages)
 * 
 * numImages must be divisible by B_X*imgsPerThread
 * numFilters must be divisible by B_Y*filtersPerThread
 */

template<int B_Y, int B_X, int imgsPerThread, int filtersPerThread, bool add, bool checkCaseBounds>
__global__ void kLocalMaxUndo2(float* maxGrads, float* _maxInds, float* target, const int imgSize, const int numFilters,
                              const int numImages, const int subsX, const int startX, const int strideX, const int outputsX,
                              const float scaleTargets, const float scaleOutputs) {
    const int* maxInds = (int *) _maxInds;
    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int blockPxX = blockIdx.x / numImgBlocks;
    const int blockPxY = blockIdx.y / (numFilters/(B_Y*filtersPerThread));
    
    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int blockFilterIdx = (blockIdx.y % (numFilters/(B_Y*filtersPerThread))) * B_Y * filtersPerThread;
    
    const int blockPx = blockPxY * imgSize + blockPxX;
    const int numOutputs = outputsX * outputsX;
    const int imgPixels = imgSize * imgSize;

    const int startOutputY = blockPxY - startX < subsX ? 0 : 1 + (blockPxY - startX - subsX) / strideX;
    const int endOutputY = MIN(outputsX, 1 + (blockPxY - startX) / strideX);
    const int startOutputX = blockPxX - startX < subsX ? 0 : 1 + (blockPxX - startX - subsX) / strideX;
    const int endOutputX = MIN(outputsX, 1 + (blockPxX - startX) / strideX);
    
    const int imgIdx = blockImgIdx + threadIdx.x;
    
    
    maxGrads += ((blockFilterIdx + threadIdx.y) * numOutputs) * numImages 
	+ imgIdx;
    maxInds += ((blockFilterIdx + threadIdx.y) * numOutputs) * numImages 
	+ imgIdx;
    
    target += ((blockFilterIdx + threadIdx.y) * imgPixels + blockPx) * numImages + imgIdx;
    
    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[f][i] = 0;
        }
    }
    
    if  (blockPxX >= startX && blockPxX < startX + strideX * (outputsX-1) + subsX 
         && blockPxY >= startX && blockPxY < startX + strideX * (outputsX-1) + subsX) {
        #pragma unroll
        for (int my = startOutputY; my < endOutputY; my++) {
            for (int mx = startOutputX; mx < endOutputX; mx++) {
                const int outputIdx = my * outputsX + mx;
                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                        #pragma unroll
                        for (int f = 0; f < filtersPerThread; f++) {
                            const int mind = maxInds[(f * B_Y * numOutputs + outputIdx) * numImages + i * B_X]; 
                            const float mg = maxGrads[(f * B_Y * numOutputs + outputIdx) * numImages + i * B_X];
                            prod[f][i] += (mind == blockPx) * mg;
                        }
                    }
                }
            }
        }
    }
    if (!add) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    target[f * B_Y * imgPixels * numImages + i * B_X] = prod[f][i];
                }
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    target[f * B_Y * imgPixels * numImages + i * B_X] = scaleTargets * target[f * B_Y * imgPixels * numImages + i * B_X] + scaleOutputs * prod[f][i];
                }
            }
        }
    }
}

void convLocalMaxUndo2(NVMatrix& maxGrads, NVMatrix& maxInds, NVMatrix& target,
		      int imgSize, int subsX, int startX, int strideX, int outputsX) {
    convLocalMaxUndo2(maxGrads, maxInds, target, imgSize, subsX, startX, strideX, outputsX, 0, 1);
}

/*
 * maxGrads:    (numFilters, numOutputs, numImages)
 * MaxInds:    (numFilters, numOutputs, numImages)
 * target:      (numFilters, imgPixels, numImages)
 */
void convLocalMaxUndo2( NVMatrix& maxGrads, NVMatrix& maxInds, NVMatrix& target,
		       int imgSize, int subsX, int startX, int strideX, int outputsX, float scaleTargets, float scaleOutput) {
    int outputs = outputsX * outputsX;
    int numImages = maxGrads.getNumCols();
    int numFilters = maxGrads.getNumRows() / outputs;
    int imgPixels = imgSize * imgSize;
    
    assert(maxGrads.getNumRows() == numFilters * outputs);
    assert(!target.isTrans());
    assert(!maxGrads.isTrans());
    assert(!maxInds.isTrans());
    assert(maxGrads.isContiguous());
    assert(maxInds.isContiguous());
    assert(maxGrads.isSameDims(maxInds));
    assert(numFilters % 16 == 0);
    //    assert(numImages % 128 == 0);
    
    assert(strideX <= subsX);
    
    target.resize(numFilters*imgPixels, numImages);
    assert(target.isContiguous());
    int imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
    int checkCaseBounds = numImages % (32*imgsPerThread) != 0;
    dim3 threads(32, 4);
    dim3 blocks(DIVUP(numImages,32*imgsPerThread) * imgSize, (numFilters / (4 * 2)) * imgSize);
    
    if (imgsPerThread == 4) {
        if  (checkCaseBounds) {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalMaxUndo2<4, 32, 4, 2, false, true><<<blocks, threads>>>( maxGrads.getDevData(), maxInds.getDevData(), target.getDevData(),
									     imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
            } else {
                kLocalMaxUndo2<4, 32, 4, 2, true, true><<<blocks, threads>>>( maxGrads.getDevData(), maxInds.getDevData(), target.getDevData(),
									    imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
            }
        } else {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalMaxUndo2<4, 32, 4, 2, false, false><<<blocks, threads>>>( maxGrads.getDevData(), maxInds.getDevData(), target.getDevData(),
									      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
            } else {
                kLocalMaxUndo2<4, 32, 4, 2, true, false><<<blocks, threads>>>( maxGrads.getDevData(), maxInds.getDevData(), target.getDevData(),
									     imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
            }
        }
    } else if (imgsPerThread == 2) {
        if  (checkCaseBounds) {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalMaxUndo2<4, 32, 2, 2, false, true><<<blocks, threads>>>( maxGrads.getDevData(), maxInds.getDevData(), target.getDevData(),
									     imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
            } else {
                kLocalMaxUndo2<4, 32, 2, 2, true, true><<<blocks, threads>>>( maxGrads.getDevData(), maxInds.getDevData(), target.getDevData(),
									    imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
            }
        } else {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalMaxUndo2<4, 32, 2, 2, false, false><<<blocks, threads>>>( maxGrads.getDevData(), maxInds.getDevData(), target.getDevData(),
									      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
            } else {
                kLocalMaxUndo2<4, 32, 2, 2, true, false><<<blocks, threads>>>( maxGrads.getDevData(), maxInds.getDevData(), target.getDevData(),
									     imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
            }
        }
    } else {
        if  (checkCaseBounds) {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalMaxUndo2<4, 32, 1, 2, false, true><<<blocks, threads>>>( maxGrads.getDevData(), maxInds.getDevData(), target.getDevData(),
									     imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
            } else {
                kLocalMaxUndo2<4, 32, 1, 2, true, true><<<blocks, threads>>>( maxGrads.getDevData(), maxInds.getDevData(), target.getDevData(),
									    imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
            }
        } else {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalMaxUndo2<4, 32, 1, 2, false, false><<<blocks, threads>>>( maxGrads.getDevData(), maxInds.getDevData(), target.getDevData(),
									      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
            } else {
                kLocalMaxUndo2<4, 32, 1, 2, true, false><<<blocks, threads>>>( maxGrads.getDevData(), maxInds.getDevData(), target.getDevData(),
									     imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
            }
        }
    }

    getLastCudaError("convLocalMaxUndo: kernel execution failed");
}



template<class Agg, int B_Y, int B_X, int imgsPerThread, int filtersPerThread, bool checkCaseBounds>
__global__ void kLocalMaxPool_Ind(float* imgs, float* target, float* _ind, const int imgSize, const int numFilters,
                           const int numImages, const int subsX, const int startX, const int strideX,
                           const int outputsX, Agg agg) {
    int* ind = (int *) _ind;
    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int numFilterBlocks = DIVUP(numFilters, B_Y*filtersPerThread);
    const int outputIdxX = blockIdx.x / numImgBlocks;
    const int outputIdxY = blockIdx.y / numFilterBlocks;
    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int blockFilterIdx = (blockIdx.y % numFilterBlocks) * B_Y * filtersPerThread;
    const int myFilterIdx = (blockFilterIdx + threadIdx.y*filtersPerThread);
    if (myFilterIdx >= numFilters) {
        return;
    }
    
    const int outputIdx = outputIdxY * outputsX + outputIdxX;
    const int numOutputs = outputsX * outputsX;
    const int imgPixels = imgSize * imgSize;
    
    const int startImgPxX = startX + outputIdxX * strideX;
    const int startImgPxY = startX + outputIdxY * strideX;
    const int imgIdx = blockImgIdx + threadIdx.x;
    
    imgs += myFilterIdx * imgPixels * numImages + imgIdx;
    target += (myFilterIdx * numOutputs + outputIdx) * numImages + imgIdx;
    ind += (myFilterIdx * numOutputs + outputIdx) * numImages + imgIdx;
    
    float prod[filtersPerThread][imgsPerThread];
    float prodind[filtersPerThread][imgsPerThread];
    #pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[f][i] = agg.getBaseValue(); 
        }
    }
    
    const int loopStartY = MAX(0, startImgPxY);
    const int loopStartX = MAX(0, startImgPxX);
    const int loopEndY = MIN(imgSize, startImgPxY + subsX);
    const int loopEndX = MIN(imgSize, startImgPxX + subsX);
    const int regionSize = (loopEndY - loopStartY) * (loopEndX - loopStartX);
    for (int y = loopStartY; y < loopEndY; y++) {
        for (int x = loopStartX; x < loopEndX; x++) {
            const int imgPx = y * imgSize + x;
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
			float tmp =  imgs[(f * imgPixels + imgPx) * numImages + i * B_X];
			if (prod[f][i] < tmp) {
			    
			    prod[f][i] = tmp;
			    prodind[f][i] = imgPx;
			}
                    }
                }
            }
        }
    }
    
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                target[f * numOutputs * numImages + i * B_X] = agg.output(prod[f][i], regionSize); 
                ind[f * numOutputs * numImages + i * B_X] = prodind[f][i];
            }
        }
    }
}


/*
 * Block size 16xB_X
 * blockIdx.x determines 4x4 pixel.x region, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines 4x4 pixel.y region, filter idx in batches of filtersPerThread
 * 
 * So each block does a 4x4 region for some number of images/filters.
 * 
 * threadIdx.x determines img idx
 * threadIdx.y determines pixel idx
 * 
 * imgs:        (numFilters, imgPixels, numImages)
 * target:      (numFilters, numOutputs, numImages)
 * 
 * B_X one of 8, 16, 32
 * imgsPerThread one of 1, 2, 4, 8, 16
 * 
 * B_XximgsPerThread MUST be divisible by 32.
 * Number of filters MUST be divisible by filtersPerThread.
 * 
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 * 
 * Final write-out will not be fully coalesced unless B_X is 32. But there's a lot more
 * reading than writing here, and the reading is all coalesced, so it should be OK.
 * 
 * To be used when the stride is 1 and the pooling region is fairly large.
 */
template<class Agg, int B_X, int imgsPerThread, int filtersPerThread, bool checkCaseBounds>
__global__ void kLocalMaxPool2_Ind(float* imgs, float* target, float *_ind, const int imgSize, const int numFilters,
			    const int numImages, const int subsX, const int startX,
			    const int outputsX, Agg agg) {
    int* ind = (int*) _ind;
    __shared__ float shImgs[filtersPerThread][B_X*imgsPerThread];
    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int numFilterBlocks = numFilters/(filtersPerThread);
    const int blockOutputX = 4*(blockIdx.x / numImgBlocks);
    const int blockOutputY = 4*(blockIdx.y / numFilterBlocks);
    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int blockFilterIdx = (blockIdx.y % numFilterBlocks) * filtersPerThread;
    
    //    const int blockOutputIdx = blockOutputY * outputsX + blockOutputX;
    const int numOutputs = outputsX * outputsX;
    const int imgPixels = imgSize * imgSize;
    
    const int tidx = threadIdx.y * B_X + threadIdx.x;
    const int loadY = tidx / 32, loadX = tidx % 32;
    
    const int myX = threadIdx.y % 4;
    const int myY = threadIdx.y / 4;
    
    const int myOutputIdxY = blockOutputY + myY;
    const int myOutputIdxX = blockOutputX + myX;
    const int myOutputIdx = myOutputIdxY * outputsX + myOutputIdxX;
    
    const int startImgPxX = startX + blockOutputX;
    const int startImgPxY = startX + blockOutputY;
    const int endImgPxX = startImgPxX + subsX;
    const int endImgPxY = startImgPxY + subsX;
    
    const int myStartImgPxY = startImgPxY + myY;
    const int myStartImgPxX = startImgPxX + myX;
    const int myEndImgPxY = endImgPxY + myY;
    const int myEndImgPxX = endImgPxX + myX;

    const int loopStartY = MAX(startImgPxY, 0);
    const int loopStartX = MAX(startImgPxX, 0);
    const int loopEndY = MIN(imgSize, endImgPxY + 3);
    const int loopEndX = MIN(imgSize, endImgPxX + 3);
    
    const int imgIdx = blockImgIdx + threadIdx.x;
    
    imgs += (blockFilterIdx + loadY) * imgPixels * numImages + blockImgIdx + loadX;
    target += (blockFilterIdx * numOutputs + myOutputIdx) * numImages + imgIdx;
    ind += (blockFilterIdx * numOutputs + myOutputIdx) * numImages + imgIdx;
    
    float prod[filtersPerThread][imgsPerThread];
    float prodind[filtersPerThread][imgsPerThread];
    #pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[f][i] = agg.getBaseValue(); 
        }
    }
    int regionSize = 0;
    for (int y = loopStartY; y < loopEndY; y++) {
        const bool isInY = y >= myStartImgPxY && y < myEndImgPxY ;
        for (int x = loopStartX; x < loopEndX; x++) {
            // Load a pixel
            const int px = y * imgSize + x;
            #pragma unroll
            for (int ly = 0; ly < filtersPerThread; ly += B_X/2) {
                if (filtersPerThread % (B_X/2) == 0 || ly + loadY < filtersPerThread) {
                    #pragma unroll
                    for (int lx = 0; lx < B_X*imgsPerThread; lx += 32) {
                        if (!checkCaseBounds || lx + loadX + blockImgIdx < numImages) {
                            shImgs[ly + loadY][lx + loadX] = imgs[(ly * imgPixels + px) * numImages + lx];
                        }
                    }
                }
            }
            __syncthreads();

            // Is this pixel in my region?
            if (isInY && x >= myStartImgPxX && x < myEndImgPxX) {
                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                        #pragma unroll
                        for (int f = 0; f < filtersPerThread; f++) {
			    if (prod[f][i] < shImgs[f][threadIdx.x + i * B_X]) {
				
				prod[f][i] = shImgs[f][threadIdx.x + i * B_X];
				prodind[f][i] = px;
			    }
                        }
                    }
                }
                ++regionSize;
            }
            __syncthreads();

        }
    }
    if (myOutputIdxY < outputsX && myOutputIdxX < outputsX) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    target[f * numOutputs * numImages + i * B_X] = agg.output(prod[f][i], regionSize); 
                    ind[f * numOutputs * numImages + i * B_X] = prodind[f][i];
                }
            }
        }
    }
}

/*
 * imgs:        (numFilters, imgPixels, numImages)
 * target:      (numFilters, outputs, numImages)
 */
void convLocalMaxPool_Ind(NVMatrix& images, NVMatrix& target, NVMatrix& targetInd, int numFilters,
                   int subsX, int startX, int strideX, int outputsX) {
    int numImages = images.getNumCols();
    int imgPixels = images.getNumRows() / numFilters;
    assert(images.getNumRows() == numFilters * imgPixels);
    int imgSize = int(sqrt(imgPixels));
    assert(imgSize * imgSize == imgPixels);
    
    assert(!images.isTrans());
    assert(!target.isTrans());
    assert(images.isContiguous());
    //    assert(numFilters % 4 == 0);
    //    assert(numImages % 128 == 0);
    
    int outputs = outputsX * outputsX;
    target.resize(numFilters*outputs, numImages);
    targetInd.resize(numFilters*outputs, numImages);

    MaxPooler pooler;
    if (strideX == 1 && subsX >= 6) {
        int imgsPerThread = numImages % 128 == 0 ? 8 : 4;
        int filtersPerThread = numFilters % 4 == 0 ? 4 : numFilters % 3 == 0 ? 3 : numFilters % 2 == 0 ? 2 : 1;
        int bx = 8;
        bool checkCaseBounds = numImages % (bx*imgsPerThread) != 0;
        assert((imgsPerThread * bx) % 32 == 0);
        assert(numFilters % filtersPerThread == 0);
        dim3 threads(bx, 16);
        dim3 blocks(DIVUP(outputsX, 4) * DIVUP(numImages, bx*imgsPerThread), DIVUP(outputsX, 4) * numFilters / filtersPerThread);
        if (imgsPerThread == 8) {
            if (filtersPerThread == 1) {
		if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalMaxPool2_Ind<MaxPooler, 8, 8, 1, true>, cudaFuncCachePreferShared);
                    kLocalMaxPool2_Ind<MaxPooler, 8, 8, 1, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(), targetInd.getDevData(),
									    imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalMaxPool2_Ind<MaxPooler, 8, 8, 1, false>, cudaFuncCachePreferShared);
                    kLocalMaxPool2_Ind<MaxPooler, 8, 8, 1, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(), targetInd.getDevData(),
									     imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                }
            } else if (filtersPerThread == 2) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalMaxPool2_Ind<MaxPooler, 8, 8, 2, true>, cudaFuncCachePreferShared);
                    kLocalMaxPool2_Ind<MaxPooler, 8, 8, 2, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(), targetInd.getDevData(),
									    imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalMaxPool2_Ind<MaxPooler, 8, 8, 2, false>, cudaFuncCachePreferShared);
                    kLocalMaxPool2_Ind<MaxPooler, 8, 8, 2, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(), targetInd.getDevData(),
									     imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                }
            } else if (filtersPerThread == 3) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalMaxPool2_Ind<MaxPooler, 8, 8, 3, true>, cudaFuncCachePreferShared);
                    kLocalMaxPool2_Ind<MaxPooler, 8, 8, 3, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(), targetInd.getDevData(),
									    imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalMaxPool2_Ind<MaxPooler, 8, 8, 3, false>, cudaFuncCachePreferShared);
                    kLocalMaxPool2_Ind<MaxPooler, 8, 8, 3, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(), targetInd.getDevData(),
									     imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                }
            } else if (filtersPerThread == 4) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalMaxPool2_Ind<MaxPooler, 8, 8, 4, true>, cudaFuncCachePreferShared);
                    kLocalMaxPool2_Ind<MaxPooler, 8, 8, 4, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(), targetInd.getDevData(),
									    imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalMaxPool2_Ind<MaxPooler, 8, 8, 4, false>, cudaFuncCachePreferShared);
                    kLocalMaxPool2_Ind<MaxPooler, 8, 8, 4, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(), targetInd.getDevData(),
									     imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                }
            }
        } else if (imgsPerThread == 4) {
            if (filtersPerThread == 1) {
		if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalMaxPool2_Ind<MaxPooler, 8, 4, 1, true>, cudaFuncCachePreferShared);
                    kLocalMaxPool2_Ind<MaxPooler, 8, 4, 1, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(), targetInd.getDevData(),
									    imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalMaxPool2_Ind<MaxPooler, 8, 4, 1, false>, cudaFuncCachePreferShared);
                    kLocalMaxPool2_Ind<MaxPooler, 8, 4, 1, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(), targetInd.getDevData(),
									     imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                }
            } else if (filtersPerThread == 2) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalMaxPool2_Ind<MaxPooler, 8, 4, 2, true>, cudaFuncCachePreferShared);
                    kLocalMaxPool2_Ind<MaxPooler, 8, 4, 2, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(), targetInd.getDevData(),
									    imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalMaxPool2_Ind<MaxPooler, 8, 4, 2, false>, cudaFuncCachePreferShared);
                    kLocalMaxPool2_Ind<MaxPooler, 8, 4, 2, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(), targetInd.getDevData(),
									     imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                }
            } else if (filtersPerThread == 3) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalMaxPool2_Ind<MaxPooler, 8, 4, 3, true>, cudaFuncCachePreferShared);
                    kLocalMaxPool2_Ind<MaxPooler, 8, 4, 3, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(), targetInd.getDevData(),
									    imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalMaxPool2_Ind<MaxPooler, 8, 4, 3, false>, cudaFuncCachePreferShared);
                    kLocalMaxPool2_Ind<MaxPooler, 8, 4, 3, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(), targetInd.getDevData(),
									     imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                }
            } else if (filtersPerThread == 4) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalMaxPool2_Ind<MaxPooler, 8, 4, 4, true>, cudaFuncCachePreferShared);
                    kLocalMaxPool2_Ind<MaxPooler, 8, 4, 4, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(), targetInd.getDevData(),
									    imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalMaxPool2_Ind<MaxPooler, 8, 4, 4, false>, cudaFuncCachePreferShared);
                    kLocalMaxPool2_Ind<MaxPooler, 8, 4, 4, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(), targetInd.getDevData(),
									     imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                }
            }
        }
    } else {
        
        int filtersPerThread = numFilters % 8 == 0 ? 2 : 1;
        int imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
        bool checkCaseBounds = numImages % (32*imgsPerThread) != 0;
        dim3 threads(32, 4);
        dim3 blocks(DIVUP(numImages,32*imgsPerThread) * outputsX, DIVUP(numFilters, 4 * filtersPerThread) * outputsX);
        if (imgsPerThread == 4) {
            if (filtersPerThread == 1) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalMaxPool_Ind<MaxPooler, 4, 32, 4, 1, true>, cudaFuncCachePreferL1);
                    kLocalMaxPool_Ind<MaxPooler, 4, 32, 4, 1, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(), targetInd.getDevData(),
									       imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalMaxPool_Ind<MaxPooler, 4, 32, 4, 1, false>, cudaFuncCachePreferL1);
                    kLocalMaxPool_Ind<MaxPooler, 4, 32, 4, 1, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(), targetInd.getDevData(),
										imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                }
            } else {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalMaxPool_Ind<MaxPooler, 4, 32, 4, 2, true>, cudaFuncCachePreferL1);
                    kLocalMaxPool_Ind<MaxPooler, 4, 32, 4, 2, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(), targetInd.getDevData(),
									       imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalMaxPool_Ind<MaxPooler, 4, 32, 4, 2, false>, cudaFuncCachePreferL1);
                    kLocalMaxPool_Ind<MaxPooler, 4, 32, 4, 2, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(), targetInd.getDevData(),
										imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                }
            }
        } else if (imgsPerThread == 2) {
            if (filtersPerThread == 1) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalMaxPool_Ind<MaxPooler, 4, 32, 2, 1, true>, cudaFuncCachePreferL1);
                    kLocalMaxPool_Ind<MaxPooler, 4, 32, 2, 1, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(), targetInd.getDevData(),
									       imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalMaxPool_Ind<MaxPooler, 4, 32, 2, 1, false>, cudaFuncCachePreferL1);
                    kLocalMaxPool_Ind<MaxPooler, 4, 32, 2, 1, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(), targetInd.getDevData(),
										imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                }
            } else {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalMaxPool_Ind<MaxPooler, 4, 32, 2, 2, true>, cudaFuncCachePreferL1);
                    kLocalMaxPool_Ind<MaxPooler, 4, 32, 2, 2, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(), targetInd.getDevData(),
									       imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalMaxPool_Ind<MaxPooler, 4, 32, 2, 2, false>, cudaFuncCachePreferL1);
                    kLocalMaxPool_Ind<MaxPooler, 4, 32, 2, 2, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(), targetInd.getDevData(),
										imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                }
            }
        } else {
            if (filtersPerThread == 1) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalMaxPool_Ind<MaxPooler, 4, 32, 1, 1, true>, cudaFuncCachePreferL1);
                    kLocalMaxPool_Ind<MaxPooler, 4, 32, 1, 1, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(), targetInd.getDevData(),
									       imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalMaxPool_Ind<MaxPooler, 4, 32, 1, 1, false>, cudaFuncCachePreferL1);
                    kLocalMaxPool_Ind<MaxPooler, 4, 32, 1, 1, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(), targetInd.getDevData(),
										imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                }
            } else {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalMaxPool_Ind<MaxPooler, 4, 32, 1, 2, true>, cudaFuncCachePreferL1);
                    kLocalMaxPool_Ind<MaxPooler, 4, 32, 1, 2, true><<<blocks, threads>>>(images.getDevData(), target.getDevData(), targetInd.getDevData(),
									       imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalMaxPool_Ind<MaxPooler, 4, 32, 1, 2, false>, cudaFuncCachePreferL1);
                    kLocalMaxPool_Ind<MaxPooler, 4, 32, 1, 2, false><<<blocks, threads>>>(images.getDevData(), target.getDevData(), targetInd.getDevData(),
										imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                }
            }
        }

    }

    getLastCudaError("convLocalPool: kernel execution failed");
}