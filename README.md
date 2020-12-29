
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;

![diagram](https://i.imgur.com/Nb0123s.png)



# Bayesian Image Reconstruction using Deep Generative Models

<span style="color:red"> Repository still under construction. Expected to be ready in early Jan. </span>


## News

* Nov 2020: Uploaded article pre-print to [arXiv](https://arxiv.org/abs/2012.04567).


Try BRGM on Google Colab: [link](insert-link-here)

# Pre-trained models available on [MIT Dropbox](https://www.dropbox.com/sh/0rj092juxauivzv/AABQoEfvM96u1ehzqYgQoD5Va?dl=0).

Download pre-trained models: 



# Installation with Anaconda

```
conda create -n "brgm" python=3.6.8 tensorflow-gpu==1.15.0 requests==2.22.0 Pillow==6.2.1 numpy==1.17.4 scikit-image==0.17.2
```

```
source activate brgm
```

```
conda install -c menpo opencv
```

```
conda install -c anaconda scipy
```


```
git clone https://github.com/razvanmarinescu/brgm.git 
```

# Installation with Docker


# Reconstructions with pre-trained networks

```
python recon.py recon-real-images --input=datasets/ffhq --tag=ffhq --network=dropbox:ffhq.pkl --recontype=super-resolution
```
