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
  
    struct mem_cell {
        int8_t p;
        uint64_t ts;
        bool passed;
        int otherAddr;
    };
    
    typedef boost::circular_buffer<mem_cell> mem_depth;
    typedef std::vector<mem_depth> Memory;

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
        virtual torch::Device inferenceDevice();
    };


    /* Background Activity Filter */
    class BackgroundActivityFilter : public EventDenoisor {
    private:
        int supporters;
        int deltaT;
        int distL2;
        bool usePolarity;
        dv::Memory memMatrix;
        
        // Addtional function
        int calculateDensity(dv::Event& event);

    public:
        BackgroundActivityFilter(uint16_t sizeX, uint16_t sizeY, std::tuple<int, int, int, bool> params);
        py::array_t<bool> run(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp);
    };


    /* Nearest Neighbor */
    class NearestNeighbor : public EventDenoisor {
    private:
        int supporters;
        int deltaT;
        int refractoryT;
        int distL2;
        bool usePolarity;
        dv::Memory memMatrix;
        
        // Addtional function
        int calculateDensity(dv::Event& event);

    public:
        NearestNeighbor(uint16_t sizeX, uint16_t sizeY, std::tuple<int, int, int, int, bool> params);
        py::array_t<bool> run(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp);
    };


    /* Khodamoradi Noise */
    class KhodamoradiNoise : public EventDenoisor {
    private:
        int supporters;
        int deltaT;
        dv::Memory xCols;
        dv::Memory yRows;

    public:
        KhodamoradiNoise(uint16_t sizeX, uint16_t sizeY, std::tuple<int, int> params);
        py::array_t<bool> run(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp);
    };


    /* Double Window Filter */
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

    
    /* Event Flow Filter */
    class EventFlowFilter : public EventDenoisor {
    private:
        float thres;
        int distL2;
        uint64_t deltaT;

        int stIdx;
        int edIdx;
        float calculateVelocity(dv::Event& event);

    public:
        EventFlowFilter(uint16_t sizeX, uint16_t sizeY, std::tuple<float, int, uint64_t> params);
        py::array_t<bool> run(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp);
    };


    /* Yang Noise */
    class YangNoise : public EventDenoisor {
    private:
        int deltaT;
        int distL2;
        int supporters;

        dv::Memory memMatrix;
    
        int lParam;
        int squareLParam;
        vector<int> modLparam;
        vector<int> dividedLparam;

        // Addtional function
        int calculateDensity(dv::Event& event);

    public:
        YangNoise(uint16_t sizeX, uint16_t sizeY, std::tuple<int, int, int> params);
        py::array_t<bool> run(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp);
    };


    /* Time Surface */
    class TimeSurface : public EventDenoisor {
    private:
        int distL2;
        float decay;
        float thres;
        int deltaTNeg;
        int deltaTPos;
        
        dv::Memory memPos;
        dv::Memory memNeg;
        
        // Addtional function
        double calculateTimeSurface(dv::Event& event);
    
    public:
        TimeSurface(uint16_t sizeX, uint16_t sizeY, std::tuple<float, int, float, int, int> params);
        py::array_t<bool> run(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp);
    };


    /* Multi Layer Perceptron Filter */
    class MultiLayerPerceptronFilter : public EventDenoisor {
    private:
        int distL2;
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
    
    
    /* Event Denoise Conv Neural Network */
    class EventDenoiseConvNeuralNetwork : public EventDenoisor {
    private:
        float thres;
        int distL2;
        int memDepth;
        int batchSize;
        string model_path;

        int minTime = 150;
        int maxTime = 5000000;

        int lParam;
        int memSize;
        dv::Memory memPos;
        dv::Memory memNeg;

        // Addtional function
        std::vector<float> buildInputTensor(dv::Event& event);
        float buildTimeSurface(uint64_t evTs, uint64_t nnTs);

    public:
        EventDenoiseConvNeuralNetwork(uint16_t sizeX, uint16_t sizeY, std::tuple<float, int, int, int, string> params);
        py::array_t<bool> run(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp);
    };
}

#endif
