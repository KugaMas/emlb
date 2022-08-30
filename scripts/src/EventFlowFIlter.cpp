#include "edn.hpp"
#include <math.h>
#include <vector>
#include <stdlib.h>
#include <Eigen/Dense>
#include <pybind11/pybind11.h>

/* Event Flow Filter */
edn::EventFlowFilter::EventFlowFilter(uint16_t sizeX, uint16_t sizeY, std::tuple<float, int, uint64_t> params) : EventDenoisor(sizeX, sizeY) {
	std::tie(thres, distL2, deltaT) = params;
	stIdx = 0;
	edIdx = 0;
}

py::array_t<bool> edn::EventFlowFilter::run(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp) {
	std::vector<bool> vec = edn::EventDenoisor::initialization(arrts, arrx, arry, arrp);

	for (int i = 0; i < evlen; i++)
	{
		dv::Event event(ptrts[i], ptrx[i], ptry[i], ptrp[i]);

		while (edIdx < evlen && ptrts[edIdx] <= event.ts + deltaT) {
			edIdx++;
		}

		while (stIdx < edIdx && ptrts[stIdx] <= event.ts - deltaT) {
			stIdx++;
		}

		if (calculateVelocity(event) < thres) {
			vec[i] = true;
		}

	}

    return py::cast(vec);
}

float edn::EventFlowFilter::calculateVelocity(dv::Event& event) {
	std::vector<int> nnIdx;

	for (int i = stIdx; i <= edIdx; i++) {
		if (abs(event.x - ptrx[i]) <= distL2 && abs(event.y - ptry[i]) <= distL2) {
			nnIdx.push_back(i);
		}
	}
	
	int len = nnIdx.size();
	if (len < 3) return thres;

	Eigen::MatrixXd A(len, 3);
	Eigen::VectorXd b(len, 1);
	for (int i = 0; i < len; i++) {
		A(i, 0) = (double) ptrx[nnIdx[i]];
		A(i, 1) = (double) ptry[nnIdx[i]];
		A(i, 2) = 1.0;
		b(i) = ((double) ptrts[nnIdx[i]] - event.ts) * 0.001;
	}

	Eigen::Vector3d X = A.colPivHouseholderQr().solve(b);

	return pow((pow(-1 / X[0], 2) + pow(-1 / X[1], 2)), 0.5);
}
