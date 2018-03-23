# Generative Adversarial Autoencoder Networks (GAAN)

Tensorflow implementation of Generative Adversarial Autoencoder Networks (GAAN)

## Setup

### Prerequisites
Tensorflow <br>

### Getting Started

We conduct experiments of our model with 1D/2D synthetic data, MNIST, CelebA and CIFAR-10 datasets.

#### 1D demo

In addition to GAAN, we also implement GAN, MDGAN, VAEGAN, WGAN-GP. Just changing model name in our code.

```
>> cd gaan_toy1d
>> python gaan_toy1d.py
```

1D demo references:

https://github.com/kremerj/gan <br>
http://notebooks.aylien.com/research/gan/gan_simple.html


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

Our implementation on MNIST/MNIST-1K is based on `tflib`: https://github.com/igul222/improved_wgan_training

First, downloading `tflib` put in the same folder of our mnist python code, eg. `gaan_mnist`:

```
>> cd gaan_mnist
>> python gaan_mnist.py
```

Comming soon.

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
@article{trung_2018_gaan,
  title={Generative Adversarial Autoencoder Networks},
  author={Ngoc-Trung Tran and Tuan-Anh Bui and Ngai-Man Cheung},
  journal={Arxiv},
  year={2018}
}
```
