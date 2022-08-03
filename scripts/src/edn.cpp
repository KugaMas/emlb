#include "edn.hpp"
#include <array>
#include <vector>
#include <stdlib.h>
#include <iostream>
#include <pybind11/pybind11.h>

#include <torch/torch.h>
#include <torch/script.h>

namespace py = pybind11;

std::vector<bool> edn::EventDenoisor::initialization(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp) {
    py::buffer_info bufp = arrp.request(), bufx = arrx.request(), bufy = arry.request(), bufts = arrts.request();
    assert(bufx.size == bufy.size && bufy.size == bufp.size && bufp.size == bufts.size);

    evlen = bufts.size;

    ptrts = static_cast<uint64_t *> (bufts.ptr);
    ptrx  = static_cast<uint16_t *> (bufx.ptr);
    ptry  = static_cast<uint16_t *> (bufy.ptr);
    ptrp  = static_cast<bool *> (bufp.ptr);

    std::vector<bool> vec(evlen, false);

    return vec;
}

/* Background Activity Filter */
edn::BackgroundActivityFilter::BackgroundActivityFilter(uint16_t sizeX, uint16_t sizeY, std::tuple<int, int, int, bool> params) : EventDenoisor(sizeX, sizeY) {
    std::tie(thres, rL2Norm, deltaT, usePolarity) = params;
    
    polMatrix = (int8_t*)  std::calloc(sizeX * sizeY, sizeof(int8_t));
    tsMatrix  = (uint64_t*) std::calloc(sizeX * sizeY, sizeof(uint64_t));
}

py::array_t<bool> edn::BackgroundActivityFilter::run(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp) {
    std::vector<bool> vec = edn::EventDenoisor::initialization(arrts, arrx, arry, arrp);

    for(int i = 0; i < evlen; i++) {
        dv::Event event(ptrts[i], ptrx[i], ptry[i], ptrp[i]);

        int evIdx = event.x * sizeY + event.y;
        if (calculateDensity(event) >= thres) {
            vec[i] = true;
        }
        tsMatrix[evIdx]  = event.ts;
        polMatrix[evIdx] = event.p;
    }

    return py::cast(vec);
}

int edn::BackgroundActivityFilter::calculateDensity(dv::Event& event) {
    int nCorrelated = 0;

    for (int i = event.x - rL2Norm; i <= event.x + rL2Norm; i++) {
        for (int j = event.y - rL2Norm; j <= event.y + rL2Norm; j++) {
            if (i < 0 || i >= sizeX || j < 0 || j >= sizeY) continue;

            int nnIdx = i * sizeY + j;
            if (polMatrix[nnIdx] == 0) continue;
            if (usePolarity && event.p != polMatrix[nnIdx]) continue;
            if (event.ts - tsMatrix[nnIdx] <= deltaT) nCorrelated++;
            if (nCorrelated >= thres) return nCorrelated;
        }
    }

    return nCorrelated;
}


/* Double Window Filter */
edn::DoubleWindowFilter::DoubleWindowFilter(uint16_t sizeX, uint16_t sizeY, std::tuple<int, int, int, bool> params) : EventDenoisor(sizeX, sizeY) {
    std::tie(thres, rL1Norm, memSize, DoubleMode) = params;
    memSize = DoubleMode ? memSize : memSize / 2;
    lastREvents.set_capacity(memSize);
    lastNEvents.set_capacity(memSize);
}

py::array_t<bool> edn::DoubleWindowFilter::run(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp) {
    std::vector<bool> vec = edn::EventDenoisor::initialization(arrts, arrx, arry, arrp);
    
    for(int i = 0; i < evlen; i++) {
        dv::Event event(ptrts[i], ptrx[i], ptry[i], ptrp[i]);

		if(!lastREvents.full()) {
			lastREvents.push_back(event);
			continue;
		} 
        
        if (calculateDensity(event) >= thres) {
            vec[i] = true;
            lastREvents.push_back(event);
        }
        else if (DoubleMode) {
            lastNEvents.push_back(event);
        }
    }

    return py::cast(vec);
}

int edn::DoubleWindowFilter::calculateDensity(dv::Event& event) {
    int distance;
    int nCorrelated = 0;
    for (const auto& lastR : lastREvents) {
        distance = std::abs(event.x - lastR.x) + std::abs(event.y - lastR.y);
        if (distance <= rL1Norm) nCorrelated++;
        if (nCorrelated >= thres) break;
    }

    for (const auto& lastN : lastNEvents) {
        distance = std::abs(event.x - lastN.x) + std::abs(event.y - lastN.y);
        if (distance <= rL1Norm) nCorrelated++;
        if (nCorrelated >= thres) break;
    }

    return nCorrelated;
}


/* Multi Layer Perceptron Filter */
edn::MultiLayerPerceptronFilter::MultiLayerPerceptronFilter(uint16_t sizeX, uint16_t sizeY, std::tuple<float, int, float, bool, bool, string> params) : EventDenoisor(sizeX, sizeY) {
    std::tie(thres, rL2Norm, tauTs, usePolarity, useTimestamp, model_path) = params;
    
    int square = (2 * rL2Norm + 1) * (2 * rL2Norm + 1);
    memSize = usePolarity * square + useTimestamp * square;

    polMatrix = (int8_t*)  std::calloc(sizeX * sizeY, sizeof(int8_t));
    tsMatrix  = (uint64_t*) std::calloc(sizeX * sizeY, sizeof(uint64_t));
}

py::array_t<bool> edn::MultiLayerPerceptronFilter::run(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp) {
    std::vector<bool> vec = edn::EventDenoisor::initialization(arrts, arrx, arry, arrp);

    torch::Device device(torch::kCPU);
    auto module = torch::jit::load(model_path, device);
    
    for(int i = 0; i < evlen; i++) {
        dv::Event event(ptrts[i], ptrx[i], ptry[i], ptrp[i]);
        
        // build input
        std::vector<float> patch = buildInputTensor(event);
        torch::Tensor input = torch::from_blob(patch.data(), {1, memSize}, torch::kFloat).to(device);

        // do inference
        torch::Tensor output = module.forward({input}).toTensor();
        if (output[0].to(torch::kCPU).item().toFloat() > thres) vec[i] = true;
    
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
        for (int i  = event.x - rL2Norm; i <= event.x + rL2Norm; i++) {
            for (int j  = event.y - rL2Norm; j <= event.y + rL2Norm; j++) {
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
        for (int i  = event.x - rL2Norm; i <= event.x + rL2Norm; i++) {
            for (int j  = event.y - rL2Norm; j <= event.y + rL2Norm; j++) {               
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

PYBIND11_MODULE(cdn_utils, m)
{
    m.doc() = "C++ implementation of event denoising algorithm";
    py::class_<edn::DoubleWindowFilter>(m, "dwf")
        .def(py::init<uint16_t, uint16_t, std::tuple<int, int, int, bool>>())
        .def("run", &edn::DoubleWindowFilter::run);

    py::class_<edn::MultiLayerPerceptronFilter>(m, "mlpf")
        .def(py::init<uint16_t, uint16_t, std::tuple<float, int, float, bool, bool, string>>())
        .def("run", &edn::MultiLayerPerceptronFilter::run);

    py::class_<edn::BackgroundActivityFilter>(m, "baf")
        .def(py::init<uint16_t, uint16_t, std::tuple<int, int, int, bool>>())
        .def("run", &edn::BackgroundActivityFilter::run);
}
