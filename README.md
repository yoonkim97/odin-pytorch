# ODIN: Out-of-Distribution Detector for Neural Networks (For Chest X-rays)

This is a [PyTorch](http://pytorch.org) implementation for detecting out-of-distribution examples in neural networks. The method is described in the paper [Principled Detection of Out-of-Distribution Examples in Neural Networks](https://arxiv.org/abs/1706.02690) by S. Liang, [Yixuan Li](http://www.cs.cornell.edu/~yli) and [R. Srikant](https://sites.google.com/a/illinois.edu/srikant/). 

## Models
We used the DenseNet models from: https://github.com/yoonkim97/DenseNet. 
Please make sure to run the [DenseNet](https://github.com/yoonkim97/DenseNet) repo before running this repository.

## Running the code 

### Dependencies

* CUDA 10.1
* PyTorch
* Anaconda2 or 3
* At least **one** GPU 

### Downloading In- and Out-of-Distribtion Datasets

To get the in- and out-of-distribution datasets, please run [ChestXRay-ImageFiltering](https://github.com/yoonkim97/ChestXRay-ImageFiltering) repo. 

### Running

Here is an example code reproducing the results of DenseNet model trained on **healthy** chest X-rays where **unhealthy** chest X-ray dataset is the out-of-distribution dataset. The temperature is set as 1000, and perturbation magnitude is set as 0.0014. In the **root** directory, run

```
cd code
# model: DenseNet, in-distribution: healthy, out-distribution: unhealthy
# magnitude: 0.0014, temperature 1000, gpu: 0
python main.py --nn healthy --out_dataset unhealthy --magnitude 0.0014 --temperature 1000 --gpu 0
```
**Note:** Please choose arguments according to the following. 

#### args
* **args.nn**: the name of the neural network model 
* **args.out_dataset**: the name of the out-of-distribution dataset
* **args.magnitude**: perturbation magnitude for input-preprocessing
* **args.temperature**: temperature scaling paramater
* **args.gpu**: when running models from [DenseNet](https://github.com/yoonkim97/DenseNet), please use 0

### Outputs
Here is an example of output. 

```
Neural network architecture:          	     DenseNet
In-distribution dataset:		     Healthy
Out-of-distribution dataset:       	     Unhealthy

                          Baseline         Our Method
FPR at TPR 95%:              34.8%               4.3% 
Detection error:              9.9%               4.6%
AUROC:                       95.3%              99.1%
AUPR In:                     96.4%              99.2%
AUPR Out:                    93.8%              99.1%
```
