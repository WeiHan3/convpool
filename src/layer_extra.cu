#include <helper_cuda.h>
#include <iostream>

#include <layer_kernels.cuh>
#include <layer.cuh>
#include <data.cuh>
#include <util.cuh>
#include <cudaconv2.cuh>
#include <matrix.h>
#include <cudaconv2_extra.cuh>

using namespace std;


ConvMaxPoolLayer::ConvMaxPoolLayer(ConvNet *convNet, PyObject* paramsDictConv, PyObject* paramsDictPool): ConvLayer(convNet, paramsDictConv) {
    //do nothing?
    _name = "convMaxPool";
    _mp_channels = pyDictGetInt(paramsDictPool, "channels");
    _mp_sizeX = pyDictGetInt(paramsDictPool, "sizeX");
    _mp_start = pyDictGetInt(paramsDictPool, "start");
    _mp_stride = pyDictGetInt(paramsDictPool, "stride");
    _mp_outputsX = pyDictGetInt(paramsDictPool, "outputsX");
    _mp_imgSize = pyDictGetInt(paramsDictPool, "imgSize");
    _mp_outputs = _mp_outputsX * _mp_outputsX;
}


void ConvMaxPoolLayer::bpropBiases(NVMatrix& v, PASS_TYPE passType) {
    int numCases = v.getNumCols();
    float scaleBGrad = passType == PASS_GC ? 1 : _biases->getEps() / numCases;
    if (_sharedBiases) {
        v.reshape(_numFilters, v.getNumElements() / _numFilters);
        _biases->getGrad().addSum(v, 1, 0, scaleBGrad);
        v.reshape(_numFilters * _mp_outputs, v.getNumElements() / (_numFilters * _mp_outputs));
    } else {
        _biases->getGrad().addSum(v, 1, 0, scaleBGrad);
    }    
}

void ConvMaxPoolLayer::bpropWeights(NVMatrix& v, int inpIdx, PASS_TYPE passType) {
    int numCases = v.getNumCols();

    NVMatrix& tgt = _partialSum > 0 ? _weightGradTmp : _weights[inpIdx].getGrad();
    float scaleWGrad = passType == PASS_GC ? 1 : _weights[inpIdx].getEps() / numCases;
    float scaleTargets = _weights[inpIdx].getNumUpdates() > 0 && _partialSum == 0; // ? 1 : 0;
    if (_randSparse->at(inpIdx)) {
	/*        convWeightActsSparse(_prev[inpIdx]->getActs(), v, tgt, _filterConns->at(inpIdx).dFilterConns, _imgSize->at(inpIdx), _modulesX, _modulesX,
                             _filterSize->at(inpIdx), _padding->at(inpIdx), _stride->at(inpIdx), _channels->at(inpIdx),
                             _filterChannels->at(inpIdx), _groups->at(inpIdx), _partialSum, scaleTargets, scaleWGrad);*/
	assert(false);
    } else {
        convMaxpoolWeightActs(_prev[inpIdx]->getActs(), v, _switches, tgt, _imgSize->at(inpIdx), _modulesX, _modulesX, _filterSize->at(inpIdx), _padding->at(inpIdx),
			      _stride->at(inpIdx), _channels->at(inpIdx), _groups->at(inpIdx), _partialSum, _mp_sizeX, _mp_start, _mp_stride, _mp_outputsX, scaleTargets, scaleWGrad);
    }
    if (_partialSum > 0) {
        scaleTargets = _weights[inpIdx].getNumUpdates() > 0;
        _weightGradTmp.reshape(_modules / _partialSum, _filterChannels->at(inpIdx) * _filterPixels->at(inpIdx) * _numFilters);
        _weights[inpIdx].getGrad().addSum(_weightGradTmp, 0, scaleTargets, 1);
        _weights[inpIdx].getGrad().reshape(_filterChannels->at(inpIdx) * _filterPixels->at(inpIdx), _numFilters);
    }    
}

void ConvMaxPoolLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (_randSparse->at(inpIdx)) {
        /*NVMatrix& tgt = _overSample->at(inpIdx) > 1 ? _actGradTmp : _prev[inpIdx]->getActsGrad();
        convImgActsSparse(v, *_weights[inpIdx], tgt, _filterConns->at(inpIdx).dFilterConns,
                          _imgSize->at(inpIdx), _imgSize->at(inpIdx), _modulesX, _padding->at(inpIdx), _stride->at(inpIdx),
                          _channels->at(inpIdx), _filterChannels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
        if (_overSample->at(inpIdx) > 1) {
            _actGradTmp.reshape(_overSample->at(inpIdx), _actGradTmp.getNumElements() / _overSample->at(inpIdx));
            _actGradTmp.sum(0, _prev[inpIdx]->getActsGrad());
            _prev[inpIdx]->getActsGrad().reshape(_prev[inpIdx]->getActsGrad().getNumElements() / v.getNumCols(), v.getNumCols());
        }
	*/
	assert(false);
    } else {
        convImgActs2(v, _switches, *_weights[inpIdx], _prev[inpIdx]->getActsGrad(), _imgSize->at(inpIdx), _imgSize->at(inpIdx), _modulesX,
		     _padding->at(inpIdx), _stride->at(inpIdx), _channels->at(inpIdx), _groups->at(inpIdx), _mp_sizeX, _mp_start, _mp_stride, _mp_outputsX, scaleTargets, 1);
    }

}
    
void ConvMaxPoolLayer::truncBwdActs() {
    ConvLayer::truncBwdActs();
    if (_conserveMem) {
	_switches.truncate();
    }
}

void ConvMaxPoolLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (_randSparse->at(inpIdx)) {
	/*
        convFilterActsSparse(*_inputs[inpIdx], *_weights[inpIdx], getActs(), _filterConns->at(inpIdx).dFilterConns,
                             _imgSize->at(inpIdx), _modulesX, _modulesX, _padding->at(inpIdx), _stride->at(inpIdx), _channels->at(inpIdx),
                             _filterChannels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
	*/
	assert(false);
    } else {
        convMaxPoolFilterActs(*_inputs[inpIdx], *_weights[inpIdx], getActs(), _switches, _imgSize->at(inpIdx), _modulesX, _modulesX, _padding->at(inpIdx),
			      _stride->at(inpIdx), _channels->at(inpIdx), _groups->at(inpIdx), _mp_sizeX, _mp_start, _mp_stride, _mp_outputsX);
    }
    
    if (scaleTargets == 0) {
        if (_sharedBiases) {
            getActs().reshape(_numFilters, getActs().getNumElements() / _numFilters);
            getActs().addVector(_biases->getW());
            getActs().reshape(_numFilters * _mp_outputs, getActs().getNumElements() / (_numFilters * _mp_outputs));
        } else {
	    assert(false);
            getActs().addVector(_biases->getW());
        }
    }
    
}
/*
void ConvMaxPoolLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    //    printf("!!!!!, shouldn't be here\n");
    //    printf("enter fpropAct convmax\n");
    
    convMaxPoolFilterActs(*_inputs[inpIdx], *(convLayer._weights[0]), *_outputs, _switches, convLayer._imgSize->at(0), convLayer._modulesX, convLayer._modulesX, convLayer._padding->at(0), \
			   convLayer._stride->at(0), convLayer._channels->at(0), convLayer._groups->at(0), poolLayer._sizeX, poolLayer._start, poolLayer._stride, poolLayer._outputsX);

    //    printf("after conv convmax\n");
    _outputs->reshape(convLayer._numFilters, _outputs->getNumElements() / convLayer._numFilters);
    _outputs->addVector(convLayer._biases->getW());

    assert(_outputs->getNumElements() == poolLayer.getActs().getNumElements());
    _outputs->reshape(poolLayer.getActs().getNumRows(), poolLayer.getActs().getNumCols());
    _outputs->equals(poolLayer.getActs());
    if (_outputs->sum() != _outputs->getNumElements()) {
	//	_outputs->apply(ReluNeuron::ReluOperator());
	//	poolLayer.getActs().apply(ReluNeuron::ReluOperator());
	
		printf("ref matrix %f, %f\n", poolLayer.getActs().max(), poolLayer.getActs().min());
		printf("new matrix %f, %f\n", _outputs->max(), _outputs->min());
		_outputs->subtract(poolLayer.getActs());
		printf("diff matrix %f, %f\n", _outputs->max(), _outputs->min());
	assert(false);
	//	assert(false);
    }
    //    printf("leave fpropAct convmax\n");
}
void ConvMaxPoolLayer::fprop(NVMatrixV& v, PASS_TYPE passType) {
    //    printf("entering fprop\n");
    assert(v.size() == _prev.size());
    _trans = false;
    _inputs.clear();
    _inputs.insert(_inputs.begin(), v.begin(), v.end());
    _outputs = _actsTarget < 0 ? _outputs : _inputs[_actsTarget];
    _rcvdFInputs = _prev.size();

    for (NVMatrixV::iterator it = v.begin(); it != v.end(); ++it) {
        (*it)->transpose(_trans);
    }
    getActs().transpose(_trans);
    
    // reference method
    cudaThreadSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    convLayer.fprop(v, passType);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float reftime;
    cudaEventElapsedTime(&reftime, start, stop);
    // new method
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    assert(v.size() == _prev.size());
    _trans = false;
    _inputs.clear();
    _inputs.insert(_inputs.begin(), v.begin(), v.end());
    _outputs = _actsTarget < 0 ? _outputs : _inputs[_actsTarget];
    _rcvdFInputs = _prev.size();

    for (NVMatrixV::iterator it = v.begin(); it != v.end(); ++it) {
        (*it)->transpose(_trans);
    }
    getActs().transpose(_trans);
    
    // First do fprop on the input whose acts matrix I'm sharing, if any
    
    fpropActs(0, 0, passType);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float newtime;
    cudaEventElapsedTime(&newtime, start, stop);

    printf("Fprop: Time for the kernel %f ,  %f ms\n", reftime, newtime);
}
void ConvMaxPoolLayer::bprop(PASS_TYPE passType) {
    poolLayer.incRcvdBInputs();
    cudaThreadSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    getActsGrad().transpose(_trans);
    for (int i = 0; i < _prev.size(); i++) {
        _prev[i]->getActs().transpose(_trans);
        _prev[i]->getActsGrad().transpose(_trans);
    }
    getActs().transpose(_trans);

    poolLayer.bpropActs(poolLayer.getActsGrad(), 0, 0, passType);
    convLayer.bpropCommon(convLayer.getActsGrad(), passType);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float reftime;
    cudaEventElapsedTime(&reftime, start, stop);
    
    // new method
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    NVMatrix &v = getActsGrad();
    int numCases = v.getNumCols();

    NVMatrix& tgt =  _weightsGradTmp;
    float scaleWGrad = convLayer._weights[0].getEps() / numCases;
    float scaleTargets = 0; // ? 1 : 0;
    int inpIdx = 0;

    v.transpose(_trans);
    for (int i = 0; i < _prev.size(); i++) {
        _prev[i]->getActs().transpose(_trans);
        _prev[i]->getActsGrad().transpose(_trans);
    }
    getActs().transpose(_trans);

    convMaxpoolWeightActs(_prev[0]->getActs(), v, _switches, tgt, convLayer._imgSize->at(inpIdx), convLayer._modulesX, convLayer._modulesX, convLayer._filterSize->at(inpIdx), convLayer._padding->at(inpIdx), convLayer._stride->at(inpIdx), convLayer._channels->at(inpIdx), convLayer._groups->at(inpIdx), convLayer._partialSum, poolLayer._sizeX, poolLayer._start, poolLayer._stride, poolLayer._outputsX, scaleTargets, scaleWGrad);
  

    _weightsGradTmp.reshape(convLayer._modules / convLayer._partialSum, convLayer._filterChannels->at(inpIdx) * convLayer._filterPixels->at(inpIdx) * convLayer._numFilters);
    NVMatrix diff;
    _weightsGradTmp.subtract(convLayer._weightGradTmp, diff);
    diff.apply(NVMatrixOps::Abs());
    
    if (diff.sum() >0) {
	printf("sum diff %f\n", diff.sum());
	//	_outputs->apply(ReluNeuron::ReluOperator());
	//	poolLayer.getActs().apply(ReluNeuron::ReluOperator());
	
	//	printf("switch matrix %f, %f, %f\n", _switches.max(), _switches.min(), _switches.mean());
	printf("new matrix %f, %f\n", _weightsGradTmp.max(), _weightsGradTmp.min());
	printf("diff matrix %f, %f\n", diff.max(), diff.min());
	//	_outputs->subtract(poolLayer.getActs());
	//	printf("diff matrix %f, %f\n", _outputs->max(), _outputs->min());
	//	assert(false);
	//	assert(false);
    }
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float newtime;
    cudaEventElapsedTime(&newtime, start, stop);

    printf("BProp: Time for the kernel %f ,  %f ms\n", reftime, newtime);
    poolLayer.bprop(passType);

/*
    if (_partialSum > 0) {
        scaleTargets = _weights[inpIdx].getNumUpdates() > 0;
        _weightGradTmp.reshape(_modules / _partialSum, _filterChannels->at(inpIdx) * _filterPixels->at(inpIdx) * _numFilters);
        _weights[inpIdx].getGrad().addSum(_weightGradTmp, 0, scaleTargets, 1);
        _weights[inpIdx].getGrad().reshape(_filterChannels->at(inpIdx) * _filterPixels->at(inpIdx), _numFilters);
    }

}
    */


    
// Do nothing if this layer has no weights
/*
void ConvMaxPoolLayer::updateWeights() {
    //convLayer.updateWeights();
}

void ConvMaxPoolLayer::copyToCPU() {
    //convLayer.copyToCPU();
}

void ConvMaxPoolLayer::copyToGPU()  {
    // convLayer.copyToGPU();
}
*/