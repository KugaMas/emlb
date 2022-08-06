# EMLB

## Prerequisites

To ensure the execution of our program, you need to download additional [pybind11](https://github.com/pybind/pybind11) and [libtorch](https://pytorch.org/) libraries to help our cpp file compile. If you have installed them before, you can skip this step and directly modify the reference of `CMakeLists.txt`.

```
cd EMLB
mkdir extern
```

**pybind11**

```
# install necessary compile environment
sudo apt-get install python3-dev
sudo apt-get install libboost-all-dev

# clone pybind11 in our repository, or you can put them in another path
git submodule add -b stable https://github.com/pybind/pybind11 extern/pybind11
```

**libtorch**
```
cd extern

wget https://download.pytorch.org/libtorch/cu116/libtorch-shared-with-deps-1.12.0%2Bcu116.zip
unzip libtorch-shared-with-deps-1.12.0+cu116.zip

```

## Requirement
+ python = 3.8
+ cuda = 11.6

you can use the following order to install relevant packages in you `Conda Env` environment. 

```
pip install -r requirements.txt
```


## Data preparation
**Layout** After downloading or collecting a new dataset, you need to rearrange your dataset in the following struture such as `/Dataset/Subclass/Sequences.*`, an example is shown as below. 

```
Datasets/
├── D-CEND
│   ├── Architecture
│   │   ├── Architecture-ND00-1.aedat4
│   │   ├── Architecture-ND00-2.aedat4
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

# Citation

If you use any of this code, please cite: 

**KNoise** Khodamoradi, Alireza, and Ryan Kastner. "$ O (N) $ O (N)-Space Spatiotemporal Filter for Reducing Noise in Neuromorphic Vision Sensors." IEEE Transactions on Emerging Topics in Computing 9.1 (2018): 15-23.

> @article{khodamoradi2018n,  
> title={$ O (N) $ O (N)-Space Spatiotemporal Filter for Reducing Noise in Neuromorphic Vision Sensors},  
> author={Khodamoradi, Alireza and Kastner, Ryan},  
> journal={IEEE Transactions on Emerging Topics in Computing},  
> year={2018},  
> publisher={IEEE}  
}

**DWF & MLPF** Guo, Shasha, and Tobi Delbruck. "Low Cost and Latency Event Camera Background Activity Denoising." IEEE Transactions on Pattern Analysis and Machine Intelligence (2022).

> @article{guo2022low,  
> title={Low Cost and Latency Event Camera Background Activity Denoising},  
> author={Guo, Shasha and Delbruck, Tobi},  
> journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},  
> year={2022},  
> publisher={IEEE}  
> }

**EDnCNN** Baldwin, R., et al. "Event probability mask (epm) and event denoising convolutional neural network (edncnn) for neuromorphic cameras." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.
> @inproceedings{baldwin2020event,  
> title={Event probability mask (epm) and event denoising convolutional neural network (edncnn) for neuromorphic cameras},  
> author={Baldwin, R and Almatrafi, Mohammed and Asari, Vijayan and Hirakawa, Keigo},  
> booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},  
> pages={1701--1710},  
> year={2020}  
> }
