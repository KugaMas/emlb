#include "edn.hpp"
#include <vector>
#include <stdlib.h>
#include <pybind11/pybind11.h>

/* Khodamoradi Noise */
edn::KhodamoradiNoise::KhodamoradiNoise(uint16_t sizeX, uint16_t sizeY, std::tuple<int, int> params) : EventDenoisor(sizeX, sizeY) {
	std::tie(supporters, deltaT) = params;
	xCols = dv::Memory(sizeX, dv::mem_depth(1));
	yRows = dv::Memory(sizeY, dv::mem_depth(1));
}

py::array_t<bool> edn::KhodamoradiNoise::run(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp) {
	std::vector<bool> vec = edn::EventDenoisor::initialization(arrts, arrx, arry, arrp);

	for (int i = 0; i < evlen; i++) {
		dv::Event event(ptrts[i], ptrx[i], ptry[i], ptrp[i]);

		size_t support = 0;
		bool xAddrMinusOne 	= (event.x > 0);
		bool xAddrPlusOne 	= (event.x < (sizeX - 1));
		bool yAddrMinusOne 	= (event.y > 0);
		bool yAddrPlusOne 	= (event.y < (sizeY - 1));

		if (xAddrMinusOne) {
			struct dv::mem_cell &xPrevCell = xCols[event.x - 1].back();
			if ((event.ts - xPrevCell.ts) <= deltaT && xPrevCell.p == event.p) {
				if ((yAddrMinusOne && (xPrevCell.otherAddr == (event.y - 1))) || (xPrevCell.otherAddr == event.y) || (yAddrPlusOne && (xPrevCell.otherAddr == (event.y + 1)))) {
					support++;
				}
			}
		}

		struct dv::mem_cell &xCell = xCols[event.x].back();
		if ((event.ts - xCell.ts) <= deltaT && xCell.p == event.p) {
			if ((yAddrMinusOne && (xCell.otherAddr == (event.y - 1))) || (yAddrPlusOne && (xCell.otherAddr == (event.y + 1)))) {
				support++;
			}
		}

		if (xAddrPlusOne) {
			struct dv::mem_cell &xNextCell = xCols[event.x + 1].back();
			if ((event.ts - xNextCell.ts) <= deltaT && xNextCell.p == event.p) {
				if ((yAddrMinusOne && (xNextCell.otherAddr == (event.y - 1))) || (xNextCell.otherAddr == event.y) || (yAddrPlusOne && (xNextCell.otherAddr == (event.y + 1)))) {
					support++;
				}
			}
		}

		if (yAddrMinusOne) {
			struct dv::mem_cell &yPrevCell = yRows[event.y - 1].back();
			if ((event.ts - yPrevCell.ts) <= deltaT && yPrevCell.p == event.p) {
				if ((xAddrMinusOne && (yPrevCell.otherAddr == (event.x - 1))) || (yPrevCell.otherAddr == event.x) || (xAddrPlusOne && (yPrevCell.otherAddr == (event.x + 1)))) {
					support++;
				}
			}
		}

		struct dv::mem_cell &yCell = yRows[event.y].back();
		if ((event.ts - yCell.ts) <= deltaT && yCell.p == event.p) {
			if ((xAddrMinusOne && (yCell.otherAddr == (event.x - 1))) || (xAddrPlusOne && (yCell.otherAddr == (event.x + 1)))) {
				support++;
			}
		}

		if (yAddrPlusOne) {
			struct dv::mem_cell &yNextCell = yRows[event.y + 1].back();
			if ((event.ts - yNextCell.ts) <= deltaT && yNextCell.p == event.p) {
				if ((xAddrMinusOne && (yNextCell.otherAddr == (event.x - 1))) || (yNextCell.otherAddr == event.x) || (xAddrPlusOne && (yNextCell.otherAddr == (event.x + 1)))) {
					support++;
				}
			}
		}

		if (support >= supporters) {
			xCell.passed = true;
			yCell.passed = true;
			vec[i] = true;
		} else {
			xCell.passed = false;
			yCell.passed = false;
		}

		// Update maps.
		xCell.ts = event.ts;
		xCell.p  = event.p;
		xCell.otherAddr = event.y;

		yCell.ts = event.ts;
		yCell.p  = event.p;
		yCell.otherAddr = event.x;

	}

	return py::cast(vec);
}
