Code for experiments in [Generative Adversarial Autoencoder Networks (GAAN)](https://arxiv.org/abs/1803.08887)

## Setup

### Prerequisites
Python, Numpy, Tensorflow <br>

### Getting Started
We conduct experiments of our model with 1D/2D synthetic data, MNIST, CelebA and CIFAR-10 datasets.

#### 1D demo
In addition to GAAN, other methods, such as GAN, MDGAN, VAEGAN, WGAN-GP are provided in our code.

```
>> cd gaan_toy1d
>> python gan_toy1d.py
```
Quick video demos, you can reproduce easily these videos with our code:

[GAN](https://www.youtube.com/watch?v=eisFNXbGaNI) <br>
[WGANGP](https://www.youtube.com/watch?v=5MDBwdfD5rY) <br>
[VAEGAN](https://www.youtube.com/watch?v=587z8VBcvvQ) <br>
[GAAN](https://www.youtube.com/watch?v=IjbdMNo4m_8)

Our 1D code is based on 1D demo references:

[1] https://github.com/kremerj/gan <br>
[2] http://notebooks.aylien.com/research/gan/gan_simple.html

#### 2D synthetic data
```
>> cd gaan_toy2d
>> python gaan_toy2d.py
```

We provides three different data layouts you can test on: 'SINE' 'PLUS' 'SQUARE'. Just change the parameter `testcase` in the code `gaan_toy2d.py`. For example:
```
testcase      = 'SQUARE'
```

#### MNIST/MNIST-1K dataset
Our implementation on MNIST/MNIST-1K is using `tflib` from [here](https://github.com/igul222/improved_wgan_training). Therefore, download `tflib` and put it in the folder of `gaan_mnist` to run our code:

```
>> cd gaan_mnist
>> python gaan_mnist.py
```

#### CelebA dataset
Our implementation on CelebA is based on follow implementation: https://github.com/LynnHo/DCGAN-LSGAN-WGAN-WGAN-GP-Tensorflow (since DCGAN is collapsed with network architecture at epoch 50).

To download CelebA dataset and change `img_paths` in `gaan_celeba.py` with your correct path. For example:

```
img_paths = glob.glob('./data/img_align_celeba/*.jpg')
```

```
>> python gaan_celeba.py
```
Comming soon.

#### CIFAR-10 dataset

```
>> python gaan_cifar.py
```
Comming soon.

## Citation
If you use this code in your research, please cite our paper:

```
@article{trung2018gaan,
  title={Generative Adversarial Autoencoder Networks},
  author={Ngoc-Trung Tran and Tuan-Anh Bui and Ngai-Man Cheung},
  journal={arXiv preprint arXiv:1803.08887},
  year={2018}
}
```
