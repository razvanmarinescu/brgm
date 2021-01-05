# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib

from training import misc
from forwardModels import *

#vggDownscale = False
vggDownscale = True
#----------------------------------------------------------------------------

class Projector:
    def __init__(self, forward=None, num_steps=5000):
        self.num_steps                  = num_steps
        self.dlatent_avg_samples        = 10000
        self.initial_learning_rate      = 0.1
        self.initial_noise_factor       = 0.05
        self.lr_rampdown_length         = 0.25
        self.lr_rampup_length           = 0.05
        self.noise_ramp_length          = 0.75
        self.regularize_noise_weight    = 0
        self.l2_pixelwise_reg           = 0.001 # kicks in after 75% completion
        self.verbose                    = False
        self.clone_net                  = True
        
        ######### FORWARD #########
        self.forward                    = forward
        ######### END-FORWARD #########

        self._Gs                    = None
        self._minibatch_size        = None
        self._dlatent_avg           = None
        self._dlatent_std           = None
        self._noise_vars            = None
        self._noise_init_op         = None
        self._noise_normalize_op    = None
        self._dlatents_var          = None
        self._noise_in              = None
        self._dlatents_expr         = None
        self._images_expr           = None
        self._images_clean          = None
        self._target_images_var     = None
        self._lpips                 = None
        self._dist                  = None
        self._loss                  = None
        self._reg_sizes             = None
        self._lrate_in              = None
        self._opt                   = None
        self._opt_step              = None
        self._cur_step              = None
        

    def _info(self, *args):
        if self.verbose:
            print('Projector:', *args)

    def cosine_distance(self, latentsBLD):
        assert latentsBLD.shape[0] == 1
        latentsNormLD = tf.nn.l2_normalize(latentsBLD[0,:,:], axis=1)
        cosDistLL = 1 - tf.matmul(latentsNormLD, latentsNormLD, transpose_b=True)   
        #return tf.reduce_mean(cosDistLL)
        return tf.reduce_mean(tf.abs(cosDistLL))
        

    def set_network(self, Gs, minibatch_size=1):
        assert minibatch_size == 1
        self._Gs = Gs
        self._minibatch_size = minibatch_size
        if self._Gs is None:
            return
        if self.clone_net:
            self._Gs = self._Gs.clone()

        # Find dlatent stats.
        self._info('Finding W midpoint and stddev using %d samples...' % self.dlatent_avg_samples)
        latent_samples = np.random.RandomState(111).randn(self.dlatent_avg_samples, *self._Gs.input_shapes[0][1:])
        dlatent_samples = self._Gs.components.mapping.run(latent_samples, None)[:, :1, :] # [N, 1, 512]
        self._dlatent_avg = np.mean(dlatent_samples, axis=0, keepdims=True) # [1, 1, 512]
        self._dlatent_std = (np.sum((dlatent_samples - self._dlatent_avg) ** 2) / self.dlatent_avg_samples) ** 0.5
        self._info('std = %g' % self._dlatent_std)

        # only optimise high-level noise
        self.numLatentLayers = self._Gs.components.synthesis.input_shape[1]
        if self.numLatentLayers == 18: 
          maxRes = 128
        else:
          maxRes = 32
        
        # Find noise inputs.
        self._info('Setting up noise inputs...')
        self._noise_vars = []
        noise_init_ops = []
        noise_normalize_ops = []
        self.noise_reg = tf.Variable(0.0, dtype=tf.float32)
        while True:
            n = 'G_synthesis/noise%d' % len(self._noise_vars)
            if not n in self._Gs.vars:
                break
            w = tf.Variable(tf.random_normal(tf.shape(self._Gs.vars[n]), dtype=tf.float32))
            self._Gs.vars[n] = (1 - self.noise_reg) * self._Gs.vars[n] + self.noise_reg * w
            v = self._Gs.vars[n]
            self._noise_vars.append(w)
            # set initial noise to zero, esp. if final layers are not optimised
            noise_init_ops.append(tf.assign(w, tf.random_normal(tf.shape(v), dtype=tf.float32)))
        self._noise_init_op = tf.group(*noise_init_ops)


        # Image output graph.
        self._info('Building image output graph...')
        self._noise_in = tf.placeholder(tf.float32, [], name='noise_in')
        self.dlatent_shape = [self._minibatch_size, self.numLatentLayers, self._dlatent_avg.shape[2]]
      
        # first we optimise dlatent (moving), then we fix it later on in the optimisation
        self.reg_dlatent_fixed = tf.Variable(0, dtype=tf.float32)
        self._dlatents_expr_fixed = tf.Variable(tf.zeros(self.dlatent_shape))
        self._dlatents_expr_moving = tf.Variable(tf.random.normal(self.dlatent_shape))
        self._dlatents_expr = (1 - self.reg_dlatent_fixed) * self._dlatents_expr_moving + self.reg_dlatent_fixed * self._dlatents_expr_fixed

        self._images_clean = self._Gs.components.synthesis.get_output_for(self._dlatents_expr, randomize_noise=False)
        
        
        ######### FORWARD #########
        # apply forward model
        self._images_expr = self.forward(self._images_clean)
        ####### END-FORWARD ##########

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        # re-scale pixels from [-1,1] to [-128,128]
        proc_images_expr = (self._images_expr + 1) * (255 / 2)

        if vggDownscale:
          sh = proc_images_expr.shape.as_list()
          if sh[2] > 256:
              factor = sh[2] // 256
              proc_images_expr_down = tf.reduce_mean(tf.reshape(proc_images_expr, [-1, sh[1], sh[2] // factor, factor, sh[2] // factor, factor]), axis=[3,5])
          else:
            proc_images_expr_down = proc_images_expr

        if proc_images_expr.shape[1] == 1:
          # if output image is mono-channel, make it 3-channel
          proc_images_expr = tf.repeat(proc_images_expr, 3, axis=1)
          proc_images_expr_down = tf.repeat(proc_images_expr_down, 3, axis=1)
  
        # Loss graph.
        self._info('Building loss graph...')
        self._target_images_down_var = tf.Variable(tf.zeros(proc_images_expr_down.shape), name='target_images_down_var')
        self._target_images_var = tf.Variable(tf.zeros(proc_images_expr.shape), name='target_images_var')
        if self._lpips is None:
            self._lpips = misc.load_pkl('http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/vgg16_zhang_perceptual.pkl')

        self._dist = self._lpips.get_output_for(proc_images_expr_down, self._target_images_down_var)
        self._perceptual_loss = tf.reduce_mean(self._dist)
        self._loss = self._perceptual_loss

        # Noise regularization graph.
        self._info('Building noise regularization graph...')
        reg_loss = 0.0
        for v in self._noise_vars:
            sz = v.shape[2]
            while True:
                reg_loss += tf.reduce_mean(v * tf.roll(v, shift=1, axis=3))**2 + tf.reduce_mean(v * tf.roll(v, shift=1, axis=2))**2
                if sz <= 8:
                    break # Small enough already
                v = tf.reshape(v, [1, 1, sz//2, 2, sz//2, 2]) # Downscale
                v = tf.reduce_mean(v, axis=[3, 5])
                sz = sz // 2
        self._loss += reg_loss * self.regularize_noise_weight

        # adding L2 loss in pixel space
        self._l2_pixelwise_reg = tf.Variable(0.0, dtype=tf.float32)
        self._pixelwise_loss = self._l2_pixelwise_reg * tf.reduce_mean(tf.squared_difference(proc_images_expr, self._target_images_var))
        self._loss += self._pixelwise_loss
        

        # adding prior on w ~ N(mu, sigma) as extra loss term
        lambda_w = 100 
        self._w_loss = lambda_w * tf.reduce_mean(tf.squared_difference(self._dlatents_expr/ self._dlatent_std, self._dlatent_avg/self._dlatent_std))
        self._loss += self._w_loss

        # adding cosine distance loss
        cosine_reg = 3 # lambda_c
        #cosine_reg = 0.0
        self._cosine_loss = cosine_reg * self.cosine_distance(self._dlatents_expr)
        self._loss += self._cosine_loss
        

        # Optimizer.
        self._info('Setting up optimizer...')
        self._lrate_in = tf.placeholder(tf.float32, [], name='lrate_in')
        self._opt = dnnlib.tflib.Optimizer(learning_rate=self._lrate_in)

        noise_vars_opt = [n for n in self._noise_vars if n.shape[-1] <= maxRes]

        self._opt.register_gradients(self._loss, [self._dlatents_expr_fixed, self._dlatents_expr_moving] + noise_vars_opt + self.forward.getVars())
        self._opt_step = self._opt.apply_updates()

    def run(self, target_images):
        # Run to completion.
        self.start(target_images)
        while self._cur_step < self.num_steps:
            self.step()

        # Collect results.
        pres = dnnlib.EasyDict()
        pres.dlatents = self.get_dlatents()
        pres.noises = self.get_noises()
        pres.images = self.get_images()
        return pres

    def start(self, target_images):
        assert self._Gs is not None

        # Prepare target images.
        self._info('Preparing target images...')
        target_images = np.asarray(target_images, dtype='float32')
        target_images = (target_images + 1) * (255 / 2)
        sh = target_images.shape
        assert sh[0] == self._minibatch_size
        if vggDownscale:
          #if sh[2] > self._target_images_var.shape[2]:
          print('self._target_images_var', self._target_images_var.shape)
          print('self._target_images_down_var', self._target_images_down_var.shape)
          if self._target_images_down_var.shape[2] < self._target_images_var.shape[2]:
            print('checkpoint vgg downscale')
            factor = sh[2] // self._target_images_down_var.shape[2]
            target_images_down = np.reshape(target_images, [-1, sh[1], sh[2] // factor, factor, sh[3] // factor, factor]).mean((3, 5))
          else:
            target_images_down = target_images

        if sh[1] == 1:  
          # if mono images, make them 3-channel, so you can use VGG to compute LPIPS distance. VGG needs 3-channel images as input
          target_images_down = np.repeat(target_images_down, 3, axis=1)
          target_images = np.repeat(target_images, 3, axis=1)

        np.random.seed(3)
        dlatent_init = self._dlatent_avg + np.random.normal(scale=0.01 * self._dlatent_std)

        # Initialize optimization state.
        self._info('Initializing optimization state...')
        tflib.set_vars({self._target_images_var: target_images, self._target_images_down_var: target_images_down, self._dlatents_expr_moving: np.tile(dlatent_init, [self._minibatch_size, self.numLatentLayers, 1]), self._dlatents_expr_fixed : np.zeros(self.dlatent_shape) ,self._l2_pixelwise_reg : 0.0, self.noise_reg : 0.0, self.reg_dlatent_fixed : 0})
        self.forward.initVars()
        print('start() target_images shape', target_images.shape)

        tflib.run(self._noise_init_op)
        self._opt.reset_optimizer_state()
        self._cur_step = 0

    def step(self):
        assert self._cur_step is not None
        if self._cur_step >= self.num_steps:
            return
        if self._cur_step == 0:
            self._info('Running...')

        # Hyperparameters.
        t = self._cur_step / self.num_steps
        noise_strength = self._dlatent_std * self.initial_noise_factor * max(0.0, 1.0 - t / self.noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / self.lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / self.lr_rampup_length)
        learning_rate = self.initial_learning_rate * lr_ramp

        l2_pixelwise_reg = self.l2_pixelwise_reg
        noise_reg_curr = 0

        reg_dlatent_fixed = 0
        
        # Train.
        feed_dict = {self._noise_in: noise_strength, self._lrate_in: learning_rate, self._l2_pixelwise_reg : l2_pixelwise_reg, self.noise_reg : noise_reg_curr, self.reg_dlatent_fixed : reg_dlatent_fixed}
        _, dist_value, _loss, _pixelwise_loss, _w_loss, _cosine_loss, _perceptual_loss = tflib.run([self._opt_step, self._dist, self._loss, self._pixelwise_loss, self._w_loss, self._cosine_loss, self._perceptual_loss], feed_dict)

        # Print status.
        self._cur_step += 1
        if self._cur_step == self.num_steps or self._cur_step % 50 == 0:
            print('_loss:', _loss, flush=True)
            print('_perceptual:', _perceptual_loss, flush=True)
            print('_pixelwise:', _pixelwise_loss, flush=True)
            print('_w_loss:', _w_loss, flush=True)
            print('_cosine:', _cosine_loss, flush=True)
            self._info('%-8d%-12g%-12g' % (self._cur_step, dist_value, _loss))
        if self._cur_step == self.num_steps:
            self._info('Done.')

    def get_cur_step(self):
        return self._cur_step

    def get_dlatents(self):
        return tflib.run(self._dlatents_expr, {self._noise_in: 0})

    def get_noises(self):
        return tflib.run(self._noise_vars)

    # returns corrupted images
    def get_images(self):
        return tflib.run(self._images_expr, {self._noise_in: 0})
    
    # returns clean images
    def get_clean_images(self):
        return tflib.run(self._images_clean, {self._noise_in: 0})

#----------------------------------------------------------------------------
