#include "edn.hpp"
#include <math.h>
#include <vector>
#include <stdlib.h>
#include <pybind11/pybind11.h>

#include <torch/torch.h>
#include <torch/script.h>

/* Event Denoise Convolutional Neural Network */
edn::EventDenoiseConvNeuralNetwork::EventDenoiseConvNeuralNetwork(uint16_t sizeX, uint16_t sizeY, std::tuple<float, int, int, int, string> params) : EventDenoisor(sizeX, sizeY) {
	std::tie(thres, distL2, memDepth, batchSize, model_path) = params;
	
	lParam = distL2 * 2 + 1;
	memSize = 2 * memDepth * lParam * lParam;

	struct dv::mem_cell nanEvent;
	memPos = dv::Memory (sizeX * sizeY, dv::mem_depth(memDepth, nanEvent));
	memNeg = dv::Memory (sizeX * sizeY, dv::mem_depth(memDepth, nanEvent));
}

py::array_t<bool> edn::EventDenoiseConvNeuralNetwork::run(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp) {
	std::vector<bool> vec = edn::EventDenoisor::initialization(arrts, arrx, arry, arrp);
	
	// load pretrained module
	torch::Device device = edn::EventDenoisor::inferenceDevice();
	auto module = torch::jit::load(model_path, device);

	// perform inference
	torch::Tensor packages = torch::empty({batchSize, 2 * memDepth, lParam, lParam}).to(torch::kCPU);
	for(int i = 0; i < evlen; i++) {
		dv::Event event(ptrts[i], ptrx[i], ptry[i], ptrp[i]);

		int k = i % batchSize;

		// build packages
		packages[k] = torch::from_blob(buildInputTensor(event).data(), {2 * memDepth, lParam, lParam}, torch::kFloat);

		// obtain output
		if (k == batchSize - 1 || i == evlen - 1) {
			auto input = packages;
			torch::Tensor output = module.forward({input.to(device)}).toTensor().to(torch::kCPU);

			float* result = (float* )(output.data_ptr());
			for (int j = 0; k >= 0; k--) {
				*result++;
				if (*result++ >= thres) vec[i - k] = true;
			}
		}
	
	}

	return py::cast(vec);

}

std::vector<float> edn::EventDenoiseConvNeuralNetwork::buildInputTensor(dv::Event& event) {
	int evIdx = event.x * sizeY + event.y;
	std::vector<float> patch(memSize, 0);

	int k = 0, layerSize = memSize / (2 * memDepth);
	// as for first event at each pixel, directly save its timestamp
	for (int i  = event.x - distL2; i <= event.x + distL2; i++) {
		for (int j  = event.y - distL2; j <= event.y + distL2; j++) {
			int xn = i < 0 ? -i : (i >= sizeX ? 2 * sizeX - i - 1 : i);
			int yn = j < 0 ? -j : (j >= sizeY ? 2 * sizeY - j - 1 : j);

			int layer = 0;
			int nnIdx = xn * sizeY + yn;

			if (event.p == 1) {
				for (const auto& pos : memPos[nnIdx]) {
					patch[(layer++) * layerSize + k] = buildTimeSurface(event.ts, pos.ts);
				}
					for (const auto& neg : memNeg[nnIdx]) {
					patch[(layer++) * layerSize + k] = buildTimeSurface(event.ts, neg.ts);
				}
			} else {
				for (const auto& neg : memNeg[nnIdx]) {
					patch[(layer++) * layerSize + k] = buildTimeSurface(event.ts, neg.ts);
				}
				for (const auto& pos : memPos[nnIdx]) {
					patch[(layer++) * layerSize + k] = buildTimeSurface(event.ts, pos.ts);
				}
			}
			k++;
		}
	}

	struct dv::mem_cell updateEvent;
	updateEvent.ts = event.ts;

	if (event.p == 1) {
		memPos[evIdx].push_front(updateEvent);
	} else {
		memNeg[evIdx].push_front(updateEvent);
	}

	return patch;
};

float edn::EventDenoiseConvNeuralNetwork::buildTimeSurface(uint64_t evTs, uint64_t nnTs) {
	float dT = evTs - nnTs;
	if (nnTs == 0 || dT >= maxTime) dT = maxTime;

	dT = log(dT + 1) - log(minTime + 1);
	if (dT <= 0) return 0;

	return dT;
};
