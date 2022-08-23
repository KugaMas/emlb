#include "edn.hpp"
#include <vector>
#include <stdlib.h>
#include <pybind11/pybind11.h>

#include <torch/torch.h>
#include <torch/script.h>

/* Multi Layer Perceptron Filter */
edn::MultiLayerPerceptronFilter::MultiLayerPerceptronFilter(uint16_t sizeX, uint16_t sizeY, std::tuple<float, int, float, int, bool, bool, string> params) : EventDenoisor(sizeX, sizeY) {
	std::tie(thres, distL2, tauTs, batchSize, usePolarity, useTimestamp, model_path) = params;
    
	int square = (2 * distL2 + 1) * (2 * distL2 + 1);
	memSize = usePolarity * square + useTimestamp * square;
	
	polMatrix = (int8_t*)  std::calloc(sizeX * sizeY, sizeof(int8_t));
	tsMatrix  = (uint64_t*) std::calloc(sizeX * sizeY, sizeof(uint64_t));
}

py::array_t<bool> edn::MultiLayerPerceptronFilter::run(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp) {
	std::vector<bool> vec = edn::EventDenoisor::initialization(arrts, arrx, arry, arrp);

	// load pretrained module
	torch::Device device = edn::EventDenoisor::inferenceDevice();
	auto module = torch::jit::load(model_path, device);

	// perform inference
	auto packages = torch::empty({batchSize, memSize});
	for(int i = 0; i < evlen; i++) {
		dv::Event event(ptrts[i], ptrx[i], ptry[i], ptrp[i]);
		
		int k = i % batchSize;
		
		// build packages
		packages[k] = torch::from_blob(buildInputTensor(event).data(), {memSize}, torch::kFloat);

    	// obtain output
    	if (k == batchSize - 1 || i == evlen - 1) {
    		auto input = packages;
    		torch::Tensor output = module.forward({input.to(device)}).toTensor().to(torch::kCPU);
            
    		float* result = (float* )(output.data_ptr());
    		for (int j = 0; k >= 0; k--) {
				if (*result++ >= thres) vec[i - k] = true;
			}
    	}  
	}

  	return py::cast(vec);

}

std::vector<float> edn::MultiLayerPerceptronFilter::buildInputTensor(dv::Event& event) {
	int k = 0;
	int evIdx = event.x * sizeY + event.y;
	std::vector<float> patch(memSize, 0);

	// as for first event at each pixel, directly save its timestamp
	if (tsMatrix[evIdx] == 0) tsMatrix[evIdx] = event.ts;

	if (useTimestamp) {
		for (int i  = event.x - distL2; i <= event.x + distL2; i++) {
			for (int j  = event.y - distL2; j <= event.y + distL2; j++) {
				// outside address space
				if (i < 0 || i >= sizeX || j < 0 || j >= sizeY) {
					patch[k++] = 0;
					continue;
				}
				
				// inside address space
				else {
					auto nnbTs = tsMatrix[i * sizeY + j];
					
					// if there us no event have appeared at a NNb pixel
					if (nnbTs <= 0 || event.ts - nnbTs > tauTs) {
						patch[k++] = 0L;
						continue;
					}
					
					// calculate timestamp
					patch[k++] = 1L - ((event.ts - nnbTs) / tauTs);
				}
			}
		}
	}

	polMatrix[evIdx] = 2 * (int) event.p - 1;
	if (usePolarity) {
		for (int i  = event.x - distL2; i <= event.x + distL2; i++) {
			for (int j  = event.y - distL2; j <= event.y + distL2; j++) {               
				// outside address space
				if (i < 0 || i >= sizeX || j < 0 || j >= sizeY) {
					patch[k++] = 0L;
					continue;
				}
				// inside address space
				else {
					auto nnbTs = tsMatrix[i * sizeY + j];
					if (nnbTs <= 0 || event.ts - nnbTs > tauTs) {
						patch[k++] = 0L;
						continue;
					}
					patch[k++] = event.p;
				}
			}
		}
  	}
	tsMatrix[evIdx] = event.ts;
	
	return patch;
};
