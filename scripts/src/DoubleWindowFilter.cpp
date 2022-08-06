#include "edn.hpp"
#include <vector>
#include <stdlib.h>
#include <pybind11/pybind11.h>

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
