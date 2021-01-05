
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;

![diagram](https://i.imgur.com/Nb0123s.png)



# Bayesian Image Reconstruction using Deep Generative Models
### R. Marinescu, D. Moyer, P. Golland

<span style="color:red"> Repository still under construction. Expected to be ready in early Jan. </span>


For technical inquiries, please create a Github issue.
For other inquiries, please contact Razvan Marinescu: razvan@csail.mit.edu

## News

* Nov 2020: Uploaded article pre-print to [arXiv](https://arxiv.org/abs/2012.04567).



## Pre-trained models available on [MIT Dropbox](https://www.dropbox.com/sh/0rj092juxauivzv/AABQoEfvM96u1ehzqYgQoD5Va?dl=0).

## Requirements

Our method builds on the StyleGAN2 Tensorflow codebase, so our requirements are the same as for StyleGAN2:
* 64-bit Python 3.6 installation. We recommend Anaconda3 with numpy 1.14.3 or newer.
* TensorFlow 1.14 (Windows and Linux) or 1.15 (Linux only). TensorFlow 2.x is not supported.
On Windows you need to use TensorFlow 1.14, as the standard 1.15 installation does not include necessary C++ headers.
* One or more high-end NVIDIA GPUs with at least 12GB DRAM, NVIDIA drivers, CUDA 10.0 toolkit and cuDNN 7.5. 


## Installation from StyleGAN2 Tensorflow environment

If you already have a StyleGAN2 Tensorflow environment in Anaconda, you can clone that environment and additionally install the missing packages: 

```
# clone environment stylegan2 into brgm
conda create --name brgm --clone stylegan2
source activate brgm

# install missing packages
conda install -c menpo opencv
conda install scikit-image==0.17.2
```

## Installation from scratch with Anaconda


Create conda environment and install packages:

```
conda create -n "brgm" python=3.6.8 tensorflow-gpu==1.15.0 requests==2.22.0 Pillow==6.2.1 numpy==1.17.4 scikit-image==0.17.2

source activate brgm

conda install -c menpo opencv
conda install -c anaconda scipy

```


Clone this github repository:
```
git clone https://github.com/razvanmarinescu/brgm.git 
```

## Installation with Docker


## Image reconstruction with pre-trained StyleGAN2 generators


Super-resolution with pre-trained FFHQ generator, on a set of unseen input images (datasets/ffhq), with super-resolution factor x32. The tag argument is optional, and appends that string to the results folder: 
```
python recon.py recon-real-images --input=datasets/ffhq --tag=ffhq \
 --network=dropbox:ffhq.pkl --recontype=super-resolution --superres-factor 32
```

Inpainting with pre-trained Xray generator (MIMIC III), using mask files from masks/1024x1024/ that match the image names exactly:
```
python recon.py recon-real-images --input=datasets/xray --tag=xray \
 --network=dropbox:xray.pkl --recontype=inpaint --masks=masks/1024x1024
```

Super-resolution on brain dataset with factor x8:
```
python recon.py recon-real-images --input=datasets/brains --tag=brains \
 --network=dropbox:brains.pkl --recontype=super-resolution --superres-factor 8
```

### Running on your images
For running on your images, pass a new folder with .png/.jpg images to --input. For inpainting, you need to pass an additional masks folder to --masks, which contains a mask file for each image in the --input folder.

## Training new StyleGAN2 generators

Follow the [StyleGAN2 instructions](https://github.com/NVlabs/stylegan2) for how to train a new generator network. In short, given a folder of images , you need to first prepare a TFRecord dataset, and then run the training code:

```
python dataset_tool.py create_from_images ~/datasets/my-custom-dataset ~/my-custom-images

python run_training.py --num-gpus=8 --data-dir=datasets --config=config-e --dataset=my-custom-dataset --mirror-augment=true
```

## Trained networks


| Network      | Dataset  | Number of input images  | Training time  | Description |
| :---------- | :------------: | :----: | :-----: | :----: | :---------- |
| ffhq.pkl | FFHQ | 60,000 (90\%) | 9 days  | Network trained on the first
