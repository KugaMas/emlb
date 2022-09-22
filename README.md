# E-MLB: Multi-Level Benchmark for Event-Based Camera Denoising

This repository implements the work in the paper [E-MLB: Multi-Level Benchmark for Event-Based Camera Denoising](https:xxxx) by 

*此处贴一系列gif结果动图*


## Overview
This repository is about a simple benchmark for denoising, both including a multi-level denoising dataset, an evaluation metric and the implementation of some SOTA algorithms. More details can be found in the [paper](https:xxxx) and [video](https:xxxx). If you use any of this repository, please cite this publication as follows:

```bibtex
@Article{Tulyakov21CVPR,
  title     = {E-MLB: Multi-Level Benchmark for Event-Based Camera Denoising},
  author    = {xxxx},
  journal   = {xxxx},
  year      = {2022},
  publisher = {IEEE}
}
```

## Running the Code
### Installation

Create a new virtual environment if needed:
```
conda create -n EMLB python=3.8
conda activate EMLB
```

Then clone this repo and install the dependencies, we'll call the directory that you cloned as `${EMLB_ROOT}`:
```
cd ${EMLB_ROOT}
git clone https://github.com/KugaMas/EMLB.git
pip install -r requirements.txt
```

You can run a **DEMO** to test EMLB with the following command:
```
TODO
```
Then you will receive MESR score on your terminal as follows:

*place an example result table here*

### Dataset

We captures a brand new Event Noisy Dataset (**END**). It should be divided into [D-END](https://drive.google.com/file/d/1ZatTSewmb-j6RsrJxMWEQIE3Sm1yraK-/view?usp=sharing) (Daytime part) and [N-END](https://drive.google.com/file/d/17ZDhuYdtHui9nqJAfiYYX27omPY7Rpl9/view?usp=sharing) (Night part), which you can download directly. The dataset structure is as follows:

```
Citationedat4
│   │   ├── ...
│   ├── Bicycle
│   │   ├── Bicycle-ND00-1.aedat4
│   │   ├── ...
│   ├── ...
│   |
├── DVS NOISE20
│   ├── alley
│   │   ├── alley-1.aedat4
│   │   ├── alley-2.aedat4
│
├── ...
```

### Build your own denoising pipeline

We provide a general template to facilitate you to evaluate your own denoising algorithm, see `${EMLB_ROOT}/scripts/utils/denoisors.py`. An example code see below:

```python
class your_own_denoisor(EventDenoisors):
    def __init__(self, size, use_polarity=True, excl_hotpixel=True, param1, param2):
        super().__init__()
        self.name           = 'Template'
        self.annotation     = 'Template'
        self.use_polarity   = use_polarity
        self.excl_hotpixel  = excl_hotpixel

        self.params = {
          'param1' : param1,
          'param2' : param2
        }
    
    @abstractmethod
    def run(self, ev, fr):
        ts, x, y, p = self.pre_prosess(self, ev, size)
        # /*--------------------------------------*/
        # input your denoising implementation here
        # /*--------------------------------------*/
        return ev
```

You can also download or collect a new dataset and put them in `${EMLB_ROOT}/Datasets`, it should rearranged in the following structure: 

```
${EMLB_ROOT}/Datasets/
├── Your Dataset
│   ├── Subclass-1
│   │   ├── Sequences-1.*
│   │   ├── Sequences-2.*
│   │   ├── ...
│   ├── Subclass-2
│   │   ├── ...
│   ├── ...
```

At present, we only support reading `aedat4`, `.pkl`, `.h5` and `.txt` files. 
+ The details of `aedat4` can be checked in [here](https://gitlab.com/inivation/dv/dv-python#open-a-recording-made-with-dv). The layout of `.pkl` or `.h5` files should be similar with `.aedat4` as follows:
```
data = {
  'events': [
    t1 x1 y1 p1
    t2 x2 y2 p2
    t3 x3 y3 p3
    ...
  ],
  'frames': [
    frame1 t1
    frame2 t2
  ],
}
```

+ If you are using `.txt` as input files, please make sure they are matched format used as follows:
```
width height
t1 x1 y1 p1
t2 x2 y2 p2
t3 x3 y3 p3
...
```

### Benchmarks

To ensure the execution of denoisors, you need to download additional [pybind11](https://github.com/pybind/pybind11) and [libtorch](https://pytorch.org/) libraries to help our cpp file compile. If you have installed them before, you can skip this step and directly modify the reference of `CMakeLists.txt`.

```
cd ${EMLB_ROOT}
mkdir extern
```

install necessary compile environment
```
sudo apt-get install python3-dev
sudo apt-get install libeigen3-dev
sudo apt-get install libboost-all-dev
```

**pybind11**

```
# clone pybind11 in our repository, or you can put them in another path
git submodule add -b stable https://github.com/pybind/pybind11 extern/pybind11
```

**libtorch**
```
cd extern

wget https://download.pytorch.org/libtorch/cu116/libtorch-shared-with-deps-1.12.0%2Bcu116.zip
unzip libtorch-shared-with-deps-1.12.0+cu116.zip

```

**Compile**

TODO


## Utils

TODO

## References

If you use any denoisors of this code, please cite the corresponding papers:

**KNoise** &nbsp; Khodamoradi, Alireza, and Ryan Kastner. "$ O (N) $ O (N)-Space Spatiotemporal Filter for Reducing Noise in Neuromorphic Vision Sensors." IEEE Transactions on Emerging Topics in Computing 9.1 (2018): 15-23.

```bibtex
@article{khodamoradi2018n,  
  title     = {$ O (N) $ O (N)-Space Spatiotemporal Filter for Reducing Noise in Neuromorphic Vision Sensors},
  author    = {Khodamoradi, Alireza and Kastner, Ryan},
  journal   = {IEEE Transactions on Emerging Topics in Computing},
  year      = {2018},
  publisher = {IEEE}
}
```

**DWF & MLPF** &nbsp; Guo, Shasha, and Tobi Delbruck. "Low Cost and Latency Event Camera Background Activity Denoising." IEEE Transactions on Pattern Analysis and Machine Intelligence (2022).

```bibtex
@article{guo2022low,  
  title     = {Low Cost and Latency Event Camera Background Activity Denoising},
  author    = {Guo, Shasha and Delbruck, Tobi},
  journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year      = {2022},
  publisher = {IEEE}
}
```

**EDnCNN** &nbsp; Baldwin, R., et al. "Event probability mask (epm) and event denoising convolutional neural network (edncnn) for neuromorphic cameras." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.

```bibtex
@inproceedings{baldwin2020event,  
  title     = {Event probability mask (epm) and event denoising convolutional neural network (edncnn) for neuromorphic cameras},
  author    = {Baldwin, R and Almatrafi, Mohammed and Asari, Vijayan and Hirakawa, Keigo},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {1701--1710},
  year      = {2020}
}
```
