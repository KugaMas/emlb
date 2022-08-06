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

    struct Matrix
    {
        boost::circular_buffer<bool> polarity;
        boost::circular_buffer<uint64_t> timestamp;
        Matrix(int size) {
            polarity.set_capacity(size);
            timestamp.set_capacity(size);
        }
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

        // Addtional function
        int calculateDensity(dv::Event& event);

    public:
        BackgroundActivityFilter(uint16_t sizeX, uint16_t sizeY, std::tuple<int, int, int, bool> params);
        py::array_t<bool> run(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp);
    };

    class DoubleWindowFilter : public EventDenoisor {
    private:
        int thres;
        int rL1Norm;
        int memSize;
        bool DoubleMode;
        boost::circular_buffer<dv::Event> lastREvents;
        boost::circular_buffer<dv::Event> lastNEvents;

        // Addtional function
        int calculateDensity(dv::Event& event);

    public:
        DoubleWindowFilter(uint16_t sizeX, uint16_t sizeY, std::tuple<int, int, int, bool> params);
        py::array_t<bool> run(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp);
    };

    class MultiLayerPerceptronFilter : public EventDenoisor {
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

        // Addtional function
        std::vector<float> buildInputTensor(dv::Event& event);
    
    public:
        MultiLayerPerceptronFilter(uint16_t sizeX, uint16_t sizeY, std::tuple<float, int, float, int, bool, bool, string> params);
        py::array_t<bool> run(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp);
    };

    class EventDenoiseConvNeuralNetwork : public EventDenoisor {
    private:
        float thres;
        int rL2Norm;
        int depth;
        int batchSize;
        string model_path;

        int minTime = 150;
        int maxTime = 5000000;

        int lParam;
        int memSize;
        std::vector<dv::Matrix> matrixPos;
        std::vector<dv::Matrix> matrixNeg;

        // Addtional function
        std::vector<float> buildInputTensor(dv::Event& event);
        float buildTimeSurface(uint64_t evTs, uint64_t nnTs);

    public:
        EventDenoiseConvNeuralNetwork(uint16_t sizeX, uint16_t sizeY, std::tuple<float, int, int, int, string> params);
        py::array_t<bool> run(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp);
    };
}

#endif