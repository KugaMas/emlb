#include "edn.hpp"
#include <vector>
#include <stdlib.h>
#include <pybind11/pybind11.h>

/* Yang Noise */
edn::YangNoise::YangNoise(uint16_t sizeX, uint16_t sizeY, std::tuple<int, int, int> params) : EventDenoisor(sizeX, sizeY) {
  std::tie(supporters, distL2, deltaT) = params;
	memMatrix = dv::Memory(sizeX * sizeY, dv::mem_depth(1));

	lParam = 2 * distL2 + 1;
	squareLParam = lParam * lParam;

	modLparam = vector<int> (squareLParam);
	dividedLparam = vector<int> (squareLParam);
	for (uint32_t i = 0; i < squareLParam; i++) {
		dividedLparam[i] = i / lParam;
		modLparam[i] = i % lParam;
	}
}

py::array_t<bool> edn::YangNoise::run(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp) {
	std::vector<bool> vec = edn::EventDenoisor::initialization(arrts, arrx, arry, arrp);

	for (int i = 0; i < evlen; i++) {
		dv::Event event(ptrts[i], ptrx[i], ptry[i], ptrp[i]);

		if (calculateDensity(event) >= supporters) {
			vec[i] = true;
		}

		int evIdx = event.x * sizeY + event.y;
		struct dv::mem_cell Element = {p:event.p, ts:event.ts};
		memMatrix[evIdx].push_front(Element);
	}
	
  return py::cast(vec);
}

int edn::YangNoise::calculateDensity(dv::Event &event)
{
	int addressX = event.x - distL2 + 1; // maybe false, erase +1
	int addressY = event.y - distL2 + 1;
	auto timeToCompare = event.ts - deltaT;
	int lInfNorm{0}; // event density performed with l infinity norm instead of l1 norm
	int sum{0};
	
	if (addressX >= 0 && addressY >= 0)
	{
		for (uint32_t i = 0; i < squareLParam; i++)
		{
			uint32_t newAddressY = addressY + modLparam[i];
			uint32_t newAddressX = addressX + dividedLparam[i];
			if (newAddressX < sizeX && newAddressY < sizeY)
			{
				struct dv::mem_cell &matrixElem = memMatrix[newAddressX * sizeY + newAddressY].back();
				if (event.p == matrixElem.p)
				{
					if (event.ts - matrixElem.ts < deltaT)
					{
						if (modLparam[i] == 0)
						{
							lInfNorm = std::max(lInfNorm, sum);
							sum = 0;
						}
						sum++;
					}
				}
			}
		}
	}

	return lInfNorm;
}
