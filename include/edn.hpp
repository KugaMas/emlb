#ifndef EDN_H
#define EDN_H

#include <vector>
#include <iostream>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <boost/circular_buffer.hpp>

namespace py = pybind11;

using namespace std;

namespace dv {

    struct Event {
        uint64_t ts;
        uint16_t x;
        uint16_t y;
        bool p;

        Event(uint64_t ts_, uint16_t x_, uint16_t y_, bool p_) : ts(ts_), x(x_), y(y_), p(p_) {}
    };
}

namespace edn {
    class EventDenoisor {
    protected:
        int32_t sizeX;
        int32_t sizeY;

        uint32_t evlen;  // Length of noise events

        bool *ptrp;
        uint16_t *ptrx;
        uint16_t *ptry;
        uint64_t *ptrts;

    public:
        EventDenoisor(uint16_t sizeX, uint16_t sizeY) : sizeX(sizeX), sizeY(sizeY) {};
        virtual ~EventDenoisor() {};
        virtual std::vector<bool> initialization(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp);
    };

    class DoubleWindowFilter: public EventDenoisor {
    private:
        int thres;
        int radius;
        int memSize;
        bool DoubleMode;
        boost::circular_buffer<dv::Event> lastREvents;
        boost::circular_buffer<dv::Event> lastNEvents;
    public:
        DoubleWindowFilter(uint16_t sizeX, uint16_t sizeY, std::tuple<int, int, bool, int> params);
        py::array_t<bool> run(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp);

        // Addtional function
        int calculateDensity(dv::Event& event);
    };

    class MultiLayerPerceptronFilter: public EventDenoisor {
    private:
        int radius;
        float tauTs;
        float thres;
        bool usePolarity;
        bool useTimestamp;
        string model_path;

        int memSize;
        int8_t *polMatrix; 
        uint64_t *tsMatrix;

    public:
        MultiLayerPerceptronFilter(uint16_t sizeX, uint16_t sizeY, std::tuple<float, int, float, bool, bool, string> params);
        py::array_t<bool> run(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp);

        // Addtional function
        std::vector<float> buildInputTensor(dv::Event& event);
    };
}

#endif