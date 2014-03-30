#include<nvmatrix.cuh>
#include<cudaconv2.cuh>
#include<cudaconv2_extra.cuh>


#include<string>
#include<iostream>

using namespace std;

class EventTimer {
public:
    EventTimer() : mStarted(false), mStopped(false) {
	cudaEventCreate(&mStart);
	cudaEventCreate(&mStop);
    }
    ~EventTimer() {
	cudaEventDestroy(mStart);
	cudaEventDestroy(mStop);
    }
    void start(cudaStream_t s = 0) { cudaEventRecord(mStart, s); 
	mStarted = true; mStopped = false; }
    void stop(cudaStream_t s = 0)  { assert(mStarted);
	cudaEventRecord(mStop, s); 
	mStarted = false; mStopped = true; }
    float elapsed() {
	assert(mStopped);
	if (!mStopped) return 0; 
	cudaEventSynchronize(mStop);
	float elapsed = 0;
	cudaEventElapsedTime(&elapsed, mStart, mStop);
	return elapsed;
    }

private:
    bool mStarted, mStopped;
    cudaEvent_t mStart, mStop;
};

struct ConvSetting {
    int imgSize;
    //    int numModulesX;
    int paddingStart;
    int moduleStride;
    int numImgColors;
    int numGroups;
    
    int numFilters;
    int filterSize;
    int partialSum;
};

struct PoolSetting {
    int sizeX;
    int start;
    int stride;
    //    int outputsX;
};

float matrix_diff(NVMatrix &m1, NVMatrix &m2) {
    NVMatrix diff;
    m1.subtract(m2, diff);
    diff.apply(NVMatrixOps::Abs());
    return diff.sum();
}

int  matrix_diff_num_elt(NVMatrix &m1, NVMatrix &m2) {
    NVMatrix diff;
    m1.equals(m2, diff);
    return diff.getNumElements() - diff.sum();
}


void checkMatrix(NVMatrix &m1, NVMatrix &m2, string msg) {
    if (matrix_diff(m1, m2) > 0) {
	cout << "\t\t" << msg << " : " << matrix_diff_num_elt(m1, m2) << endl;
    }
}
void test_setting(ConvSetting cs, PoolSetting ps, int numImages) {
// testing for correctness and speed
    int numModulesX = 1 + int(ceil((-cs.paddingStart * 2 + cs.imgSize - cs.filterSize) / float(cs.moduleStride)));
    int numModules = numModulesX * numModulesX;
    int outputsX = 1 + int(ceil((numModulesX - ps.start - ps.sizeX)/float(ps.stride)));
    int filterPixels = cs.filterSize * cs.filterSize;
    int numFilterColors = cs.numImgColors / cs.numGroups;
    int filterPixel = cs.filterSize * cs.filterSize;
    int filterColor = cs.numImgColors / cs.numGroups;

    
    NVMatrix input, filters;
    NVMatrix output_ref, output_ref2;
    NVMatrix output_ref_temp;
    NVMatrix output_new;
    NVMatrix output_new_switch;
    NVMatrix output_ref_switch;
    NVMatrix inGrads;

    
    input.resize(cs.numImgColors * cs.imgSize * cs.imgSize, numImages);
    filters.resize(filterPixel * filterColor, cs.numFilters);
    
    input.randomizeGaussian(0, 100.0);
    filters.randomizeGaussian(0, 1.0);

    float reftime, newtmptime, newtime;
    EventTimer timer;
    
    inGrads.resize(outputsX * outputsX * cs.numFilters, numImages);
    
    inGrads.randomizeGaussian(0, 10);

    // reference foward
    timer.start();

    convFilterActs(input, filters, output_ref_temp, cs.imgSize, numModulesX, numModulesX, cs.paddingStart, cs.moduleStride, cs.numImgColors, cs.numGroups);
    convLocalPool(output_ref_temp, output_ref, cs.numFilters, ps.sizeX, ps.start, ps.stride, outputsX, MaxPooler());
    
    timer.stop();

    reftime = timer.elapsed();


    // convpool with temp matrix
    timer.start();
    convFilterActs(input, filters, output_ref_temp, cs.imgSize, numModulesX, numModulesX, cs.paddingStart, cs.moduleStride, cs.numImgColors, cs.numGroups);
    convLocalMaxPool_Ind(output_ref_temp, output_ref2, output_ref_switch, cs.numFilters, ps.sizeX, ps.start, ps.stride, outputsX);
    timer.stop();

    newtmptime = timer.elapsed();
    
    timer.start();
    
    convMaxPoolFilterActs(input, filters, output_new, output_new_switch, cs.imgSize, numModulesX, numModulesX, cs.paddingStart, cs.moduleStride, cs.numImgColors, cs.numGroups, ps.sizeX, ps.start, ps.stride, outputsX);

    timer.stop();
    
    newtime = timer.elapsed();


    checkMatrix(output_ref, output_new, "filterAct_new");
    printf("\tfilterAct: %f, %f, %f", reftime, newtmptime, newtime);


    NVMatrix grad_tmp_ref, grad_tmp_new;
    NVMatrix grad_ref, grad_new;

    NVMatrix &grad_ref_target = cs.partialSum > 0 ? grad_tmp_ref: grad_ref;
    NVMatrix &grad_new_target = cs.partialSum > 0 ? grad_tmp_new: grad_new;

    NVMatrix ref_tmp, ref_tmp2;

    // running the refence

    timer.start();

    convLocalMaxUndo(output_ref_temp, inGrads, output_ref, ref_tmp, ps.sizeX, ps.start, ps.stride, outputsX);
    convWeightActs(input, ref_tmp, grad_ref_target, cs.imgSize, numModulesX, numModulesX, cs.filterSize, cs.paddingStart, cs.moduleStride, cs.numImgColors, cs.numGroups, cs.partialSum);
    if (cs.partialSum > 0) {
	grad_tmp_ref.reshape(numModulesX*numModulesX /cs.partialSum, numFilterColors * filterPixels * cs.numFilters);
	grad_ref.addSum(grad_tmp_ref, 0, 0, 1);
	grad_ref.reshape(numFilterColors * filterPixels, cs.numFilters);
    }

    timer.stop();
    reftime = timer.elapsed();
    // running convpool with temp matrix

    timer.start();

    convLocalMaxUndo2(inGrads, output_new_switch, ref_tmp2, numModulesX, ps.sizeX, ps.start, ps.stride, outputsX);
    convWeightActs(input, ref_tmp, grad_ref_target, cs.imgSize, numModulesX, numModulesX, cs.filterSize, cs.paddingStart, cs.moduleStride, cs.numImgColors, cs.numGroups, cs.partialSum);
    if (cs.partialSum > 0) {
	grad_tmp_ref.reshape(numModulesX*numModulesX /cs.partialSum, numFilterColors * filterPixels * cs.numFilters);
	grad_ref.addSum(grad_tmp_ref, 0, 0, 1);
	grad_ref.reshape(numFilterColors * filterPixels, cs.numFilters);
    }

    timer.stop();

    newtmptime = timer.elapsed();
    
    timer.start();

    convMaxpoolWeightActs(input, inGrads, output_new_switch, grad_new_target, cs.imgSize, numModulesX, numModulesX, cs.filterSize, cs.paddingStart, cs.moduleStride, cs.numImgColors, cs.numGroups, cs.partialSum, ps.sizeX, ps.start, ps.stride, outputsX);

    if (cs.partialSum > 0) {
	grad_tmp_new.reshape(numModulesX*numModulesX /cs.partialSum, numFilterColors * filterPixels * cs.numFilters);
	grad_new.addSum(grad_tmp_new, 0, 0, 1);
	grad_new.reshape(numFilterColors * filterPixels, cs.numFilters);
    }

    timer.stop();
    newtime = timer.elapsed();

    checkMatrix(grad_new, grad_ref, "weightAct_new");


    printf("\tweightAct: %f, %f, %f", reftime, newtmptime, newtime);

    NVMatrix imgActTgt_ref, imgActTgt_new;
    /// now start imgActsTest
    
    timer.start();
    convImgActs(ref_tmp, filters, imgActTgt_ref, cs.imgSize, cs.imgSize, numModulesX, cs.paddingStart, cs.moduleStride, cs.numImgColors, cs.numGroups);

    timer.stop();
    reftime = timer.elapsed();


    timer.start();
    
    convImgActs2(inGrads, output_new_switch, filters, imgActTgt_new, cs.imgSize, cs.imgSize, numModulesX, cs.paddingStart, cs.moduleStride, cs.numImgColors, cs.numGroups, ps.sizeX, ps.start, ps.stride, outputsX);

    timer.stop();
    newtime = timer.elapsed();
    

    checkMatrix(imgActTgt_ref, imgActTgt_new, "imageAct_new");

    printf("\timageAct: %f, %f, %f\n", reftime, 0.0, newtime);

}


void test() {
    int imgsizes[] = {32, 64, 128};
    int ms[] = {1, 2, 3, 4};
    
    int color[] = {3, 12, 16, 32};

    int nf[] = {32, 64, 128};

    int fs[] = {4, 5, 6};
    
    for (int i = 0; i < sizeof(imgsizes)/sizeof(int); i++) 
	for (int j = 0; j < sizeof(ms)/sizeof(int); j++) 
	    for (int k = 0; k < sizeof(color)/sizeof(int); k++)
		for (int l = 0; l < sizeof(nf)/sizeof(int); l++)
		    for (int m = 0; m < sizeof(fs)/sizeof(int); m++) {
			ConvSetting cs = {
			    imgsizes[i], //int imgSize;
			    0,//int paddingStart;
			    ms[j], //int moduleStride;
			    color[k],//int numImgColors;
			    1, //int numGroups;    
			    nf[l],//int numFilters;
			    fs[m],//int filterSize;
			    0,// int partialSum
			};
			PoolSetting ps = {
			    2, //int sizeX;
			    0, //int start;
			    2, //int stride;
			    //	, //int outputsX;
			};
			test_setting( cs,  ps, 128);
		    }

}


int main() {
    NVMatrix::initRandom();
    test();
    return 0;
}
