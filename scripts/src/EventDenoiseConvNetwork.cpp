#include "edn.hpp"
#include <math.h>
#include <vector>
#include <stdlib.h>
#include <pybind11/pybind11.h>

#include <torch/torch.h>
#include <torch/script.h>

/* Event Denoise Convolutional Neural Network */
edn::EventDenoiseConvNeuralNetwork::EventDenoiseConvNeuralNetwork(uint16_t sizeX, uint16_t sizeY, std::tuple<float, int, int, int, string> params) : EventDenoisor(sizeX, sizeY) {
    std::tie(thres, rL2Norm, depth, batchSize, model_path) = params;
    
    lParam = rL2Norm * 2 + 1;
    memSize = 2 * depth * lParam * lParam;

    matrixPos = std::vector<dv::Matrix> (sizeX * sizeY, dv::Matrix(depth));
    matrixNeg = std::vector<dv::Matrix> (sizeX * sizeY, dv::Matrix(depth));

    for (int i = 0; i < sizeX * sizeY; i++) {
        for (int j = 0; j < depth; j++) {
            matrixPos[i].timestamp.push_back(0);
            matrixNeg[i].timestamp.push_back(0);
        }
    }

}

py::array_t<bool> edn::EventDenoiseConvNeuralNetwork::run(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp) {
    std::vector<bool> vec = edn::EventDenoisor::initialization(arrts, arrx, arry, arrp);

    // choose inference device
    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        device_type = torch::kCUDA;
    } else {
        device_type = torch::kCPU;
        std::cout << "Warning: you are using cpu to inference, please check your graphic cards!" << std::endl;
    }
    torch::Device device = torch::Device(device_type);

    // load pretrained module
    auto module = torch::jit::load(model_path, device);

    // perform inference
    torch::Tensor packages = torch::empty({batchSize, 2 * depth, lParam, lParam}).to(torch::kCPU);
    for(int i = 0; i < evlen; i++) {
        dv::Event event(ptrts[i], ptrx[i], ptry[i], ptrp[i]);

        int k = i % batchSize;

        // build packages
        packages[k] = torch::from_blob(buildInputTensor(event).data(), {2 * depth, lParam, lParam}, torch::kFloat);

        // // obtain output
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

    int k = 0, layerSize = memSize / (2 * depth);
    // as for first event at each pixel, directly save its timestamp
    for (int i  = event.x - rL2Norm; i <= event.x + rL2Norm; i++) {
        for (int j  = event.y - rL2Norm; j <= event.y + rL2Norm; j++) {
            int xn = i < 0 ? -i : (i >= sizeX ? 2 * sizeX - i - 1 : i);
            int yn = j < 0 ? -j : (j >= sizeY ? 2 * sizeY - j - 1 : j);

            int layer = 0;
            int nnIdx = xn * sizeY + yn;

            if (event.p == 1) {
                for (const auto& posTs : matrixPos[nnIdx].timestamp) {
                    patch[(layer++) * layerSize + k] = buildTimeSurface(event.ts, posTs);
                }
                for (const auto& negTs : matrixNeg[nnIdx].timestamp) {
                    patch[(layer++) * layerSize + k] = buildTimeSurface(event.ts, negTs);
                }
            } else {
                for (const auto& negTs : matrixNeg[nnIdx].timestamp) {
                    patch[(layer++) * layerSize + k] = buildTimeSurface(event.ts, negTs);
                }
                for (const auto& posTs : matrixPos[nnIdx].timestamp) {
                    patch[(layer++) * layerSize + k] = buildTimeSurface(event.ts, posTs);
                }
            }

            k++;
        }
    }

    if (event.p == 1) {
        matrixPos[evIdx].timestamp.push_front(event.ts);
    } else {
        matrixNeg[evIdx].timestamp.push_front(event.ts);
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