#include "edn.hpp"
#include <vector>
#include <stdlib.h>
#include <pybind11/pybind11.h>

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