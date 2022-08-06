#ifndef EDN_H
#define EDN_H

#include <vector>
#include <iostream>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <boost/circular_buffer.hpp>
#include <torch/torch.h>
#include <torch/script.h>

namespace py = pybind11;

using namespace std;

namespace dv {

    struct Event {
        uint64_t ts;
        int16_t x;
        int16_t y;
        int8_t  p;

        Event(uint64_t ts_, uint16_t x_, uint16_t y_, bool p_) : ts(ts_), x(x_), y(y_), p(2 * p_ - 1) {}
    };

    struct matrix
    {
        uint64_t timestamp;
        bool polarity;
        matrix() : timestamp(0), polarity(false) {}
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

    class BackgroundActivityFilter : public EventDenoisor {
    private:
        int thres;
        int deltaT;
        int rL2Norm;
        bool usePolarity;
        int8_t *polMatrix; 
        uint64_t *tsMatrix;

    public:
        BackgroundActivityFilter(uint16_t sizeX, uint16_t sizeY, std::tuple<int, int, int, bool> params);
        py::array_t<bool> run(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp);

        // Addtional function
        int calculateDensity(dv::Event& event);
    };

    class DoubleWindowFilter: public EventDenoisor {
    private:
        int thres;
        int rL1Norm;
        int memSize;
        bool DoubleMode;
        boost::circular_buffer<dv::Event> lastREvents;
        boost::circular_buffer<dv::Event> lastNEvents;
    public:
        DoubleWindowFilter(uint16_t sizeX, uint16_t sizeY, std::tuple<int, int, int, bool> params);
        py::array_t<bool> run(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp);

        // Addtional function
        int calculateDensity(dv::Event& event);
    };

    class MultiLayerPerceptronFilter: public EventDenoisor {
    private:
        int rL2Norm;
        float tauTs;
        float thres;
        int batchSize;
        bool usePolarity;
        bool useTimestamp;
        string model_path;

        int memSize;
        int8_t *polMatrix; 
        uint64_t *tsMatrix;

    public:
        MultiLayerPerceptronFilter(uint16_t sizeX, uint16_t sizeY, std::tuple<float, int, float, int, bool, bool, string> params);
        py::array_t<bool> run(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp);

        // Addtional function
        std::vector<float> buildInputTensor(dv::Event& event);
    };
}

#endif