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

torch::Device edn::EventDenoisor::inferenceDevice() {
	// choose inference device
    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        device_type = torch::kCUDA;
    } else {
        device_type = torch::kCPU;
        std::cout << "Warning: you are using cpu to inference, please check your graphic cards!" << std::endl;
    }
    torch::Device device = torch::Device(device_type);
	return device;
}

PYBIND11_MODULE(cdn_utils, m)
{
	m.doc() = "C++ implementation of event denoising algorithm";
	py::class_<edn::BackgroundActivityFilter>(m, "baf")
		.def(py::init<uint16_t, uint16_t, std::tuple<int, int, int, bool>>())
		.def("run", &edn::BackgroundActivityFilter::run);

	py::class_<edn::NearestNeighbor>(m, "nn")
		.def(py::init<uint16_t, uint16_t, std::tuple<int, int, int, int, bool>>())
		.def("run", &edn::NearestNeighbor::run);

	py::class_<edn::KhodamoradiNoise>(m, "knoise")
		.def(py::init<uint16_t, uint16_t, std::tuple<int, int>>())
		.def("run", &edn::KhodamoradiNoise::run);

	py::class_<edn::DoubleWindowFilter>(m, "dwf")
		.def(py::init<uint16_t, uint16_t, std::tuple<int, int, int, bool>>())
		.def("run", &edn::DoubleWindowFilter::run);
		
	py::class_<edn::EventFlowFilter>(m, "evflow")
		.def(py::init<uint16_t, uint16_t, std::tuple<float, int, uint64_t>>())
		.def("run", &edn::EventFlowFilter::run);

	py::class_<edn::YangNoise>(m, "ynoise")
		.def(py::init<uint16_t, uint16_t, std::tuple<int, int, int>>())
		.def("run", &edn::YangNoise::run);

	py::class_<edn::TimeSurface>(m, "timesurface")
		.def(py::init<uint16_t, uint16_t, std::tuple<float, int, float, int, int>>())
		.def("run", &edn::TimeSurface::run);

	py::class_<edn::MultiLayerPerceptronFilter>(m, "mlpf")
		.def(py::init<uint16_t, uint16_t, std::tuple<float, int, float, int, bool, bool, string>>())
		.def("run", &edn::MultiLayerPerceptronFilter::run);

	py::class_<edn::EventDenoiseConvNeuralNetwork>(m, "edncnn")
		.def(py::init<uint16_t, uint16_t, std::tuple<float, int, int, int, string>>())
		.def("run", &edn::EventDenoiseConvNeuralNetwork::run);
}
