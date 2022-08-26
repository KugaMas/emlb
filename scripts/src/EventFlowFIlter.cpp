#include "edn.hpp"
#include <vector>
#include <stdlib.h>
#include "KDTree.hpp"
#include <Eigen/Dense>
#include <pybind11/pybind11.h>


/* Event Flow Filter */
edn::EventFlowFilter::EventFlowFilter(uint16_t sizeX, uint16_t sizeY, std::tuple<float, float> params) : EventDenoisor(sizeX, sizeY) {
	std::tie(thres, distL2) = params;
}

py::array_t<bool> edn::EventFlowFilter::run(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp) {
	std::vector<bool> vec = edn::EventDenoisor::initialization(arrts, arrx, arry, arrp);

	pointVec points = KDPoint();
	KDTree tree(points);

	for (int i = 0; i < evlen; i++)
	{
		auto res = tree.neighborhood_points(points[i], distL2);

		if (res.size() >= 3)
		{
			Eigen::MatrixXd A(res.size(), 3);
			Eigen::VectorXd b(res.size(), 1);
			int r = 0;
			for (point_t sub_res : res)
			{
				int c = 0;
				for (double element : sub_res)
				{
					if (c < 2)
					{
						A(r, c) = element;
					}
					else
					{
						A(r, c) = 1.0;
						b(r) = element;
					}
					c++;
				}
				r++;
			}

			Eigen::Vector3d X = A.colPivHouseholderQr().solve(b);
			if (pow((pow(-1 / X[0], 2) + pow(-1 / X[1], 2)), 0.5) <= thres)
			{
				vec[i] = true;
			}
		}
	}

    return py::cast(vec);
}

pointVec edn::EventFlowFilter::KDPoint() {
    pointVec points;
    
	for (int i = 0; i < evlen; i++)
	{
		point_t pt = {(double)ptrx[i], (double)ptry[i], (double) (ptrts[i] - ptrts[0]) * 0.001};
		points.push_back(pt);
	}

    return points;
}

