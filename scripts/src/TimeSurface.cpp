#include "edn.hpp"
#include <vector>
#include <stdlib.h>
#include <pybind11/pybind11.h>


/* Time Surface */
edn::TimeSurface::TimeSurface(uint16_t sizeX, uint16_t sizeY, std::tuple<float, int, float, int, int> params) : EventDenoisor(sizeX, sizeY) {
	std::tie(thres, distL2, decay, deltaTNeg, deltaTPos) = params;
    
	memPos = dv::Memory(sizeX * sizeY, dv::mem_depth(1));
	memNeg = dv::Memory(sizeX * sizeY, dv::mem_depth(1));
}

py::array_t<bool> edn::TimeSurface::run(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp) {
	std::vector<bool> vec = edn::EventDenoisor::initialization(arrts, arrx, arry, arrp);

	for(int i = 0; i < evlen; i++) {
		dv::Event event(ptrts[i], ptrx[i], ptry[i], ptrp[i]);
        int evIdx = event.x * sizeY + event.y;

        bool pass = false;
		if (calculateTimeSurface(event) <= thres) {
            pass = true;
			if (deltaTPos <= 0) vec[i] = true;
            else {
                auto prev = event.p == 1 ? memPos[evIdx].back():memNeg[evIdx].back();
                if (prev.passed && prev.otherAddr != 0 && prev.otherAddr < evlen) {
                    if (event.ts - prev.ts > deltaTPos) {
                        vec[prev.otherAddr] = true;
                    }
                }
            }
		}

		struct dv::mem_cell Element = {event.p, event.ts, pass, i};
        if (event.p == 1) {
            memPos[evIdx].push_front(Element);
        } else {
            memNeg[evIdx].push_front(Element);
        }
	}

	return py::cast(vec);
}

double edn::TimeSurface::calculateTimeSurface(dv::Event& event) {
    int nCorrelated = 0;
	double_t diff = 0;

    if (event.p == 1) {
        auto prev = memPos[event.x * sizeY + event.y].back();
        if (prev.ts != 0 && event.ts - prev.ts <= deltaTNeg) return 1;

        for (int i = event.x - distL2; i <= event.x + distL2; i++) {
            for (int j = event.y - distL2; j <= event.y + distL2; j++) {
                if (i < 0 || i >= sizeX || j < 0 || j >= sizeY) continue;

                int nnIdx = i * sizeY + j;
                struct dv::mem_cell &neighbor = memPos[nnIdx].back();

                if (neighbor.p == 0) continue;
                diff += 1 - exp(-(double)(event.ts - neighbor.ts) / decay);
                nCorrelated++;
            }
        }
    } else {
        auto prev = memNeg[event.x * sizeY + event.y].back();
        if (prev.ts != 0 && event.ts - prev.ts <= deltaTNeg) return 1;

        for (int i = event.x - distL2; i <= event.x + distL2; i++) {
            for (int j = event.y - distL2; j <= event.y + distL2; j++) {
                if (i < 0 || i >= sizeX || j < 0 || j >= sizeY) continue;

                int nnIdx = i * sizeY + j;
                struct dv::mem_cell &neighbor = memNeg[nnIdx].back();

                if (neighbor.p == 0) continue;
                diff += 1 - exp(-(double)(event.ts - neighbor.ts) / decay);
                nCorrelated++;
            }
        }
    }

	return diff / nCorrelated;
}
