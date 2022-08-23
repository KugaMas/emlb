#include "edn.hpp"
#include <vector>
#include <stdlib.h>
#include <pybind11/pybind11.h>

/* Background Activity Filter */
edn::BackgroundActivityFilter::BackgroundActivityFilter(uint16_t sizeX, uint16_t sizeY, std::tuple<int, int, int, bool> params) : EventDenoisor(sizeX, sizeY) {
	std::tie(supporters, distL2, deltaT, usePolarity) = params;

	memMatrix = dv::Memory(sizeX * sizeY, dv::mem_depth(1));
	// you can use this faster implementation, the above is only for unified interface
	// int8_t *polMatrix = (int8_t*)  	std::calloc(sizeX * sizeY, sizeof(int8_t));
	// uint64_t *tsMatrix  = (uint64_t*) std::calloc(sizeX * sizeY, sizeof(uint64_t));
}

py::array_t<bool> edn::BackgroundActivityFilter::run(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp) {
	std::vector<bool> vec = edn::EventDenoisor::initialization(arrts, arrx, arry, arrp);

	for(int i = 0; i < evlen; i++) {
		dv::Event event(ptrts[i], ptrx[i], ptry[i], ptrp[i]);

		if (calculateDensity(event) >= supporters) {
			vec[i] = true;
		}

		int evIdx = event.x * sizeY + event.y;
		struct dv::mem_cell Element = {event.p, event.ts};
		memMatrix[evIdx].push_front(Element);
	}

	return py::cast(vec);
}

int edn::BackgroundActivityFilter::calculateDensity(dv::Event& event) {
	int nCorrelated = 0;

	for (int i = event.x - distL2; i <= event.x + distL2; i++) {
		for (int j = event.y - distL2; j <= event.y + distL2; j++) {
			if (i < 0 || i >= sizeX || j < 0 || j >= sizeY) continue;

			int nnIdx = i * sizeY + j;
			struct dv::mem_cell &neighbor = memMatrix[nnIdx].back();

			if (neighbor.p == 0) continue;
			if (usePolarity && event.p != neighbor.p) continue;
			if (event.ts - neighbor.ts <= deltaT) nCorrelated++;
			if (nCorrelated >= supporters) return nCorrelated;
		}
	}

	return nCorrelated;
}
