# Code by Razvan Marinescu, razvan@csail.mit.edu
# Adapted from the StyleGAN2 code: https://github.com/NVlabs/stylegan2

import multiprocessing
import os, os.path
import cv2


import argparse
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import re
import sys

# reconstruct all the stages in the loss function until the final version. Starting from the original StyleGAN2 inversion code, we add new features

import projector # + cosine loss
#import projector_priorw as projector # + prior_w
#import projector_l2 as projector # + pixelwise L2
#import projector_wplus as projector # + W+
#import projector_nonoise as projector # + no noise
#import projector_orig as projector # original StyleGAN2 projector, but with corruption forward model
 

import pretrained_networks
from training import dataset
from training import misc

from forwardModels import *
import tensorflow as tf
import numpy as np
import skimage.io
    
runSerial = True
#runSerial = False

def constructForwardModel(recontype, imgSize, nrChannels, mask_dir, imgShort, superres_factor):
  if recontype == 'none':
    forward = ForwardNone(); forwardTrue = forward # no forward model, just image inversion

  elif recontype == 'super-resolution':
    # Create downsampling forward corruption model
    forward = ForwardDownsample(factor=superres_factor); forwardTrue = forward # res = target resolution

  elif recontype == 'inpaint':
    # Create forward model that fills in part of image with zeros (change the height/width to control the bounding box)
    forward = ForwardFillMask();
    maskFile = '%s/%s' % (mask_dir, imgShort) # mask should have same name as image 
    print('Loading mask %s' % maskFile)
    mask = skimage.io.imread(maskFile)
    mask = mask[:,:,0] == np.min(mask[:,:,0]) # need to block black color

    mask = np.reshape(mask, (1,1, mask.shape[0], mask.shape[1]))
    forward.mask = np.repeat(mask, nrChannels, axis=1)
    forwardTrue = forward 

  else:
    raise ValueError('recontype has to be either none, super-resolution, inpaint')
  
  return forward, forwardTrue

def getImgSize(Gs):
  Gs_kwargs = dnnlib.EasyDict()
  Gs_kwargs.randomize_noise = False
  Gs_kwargs.truncation_psi = 0.5
  #print(Gs.components.synthesis.resolution)

  noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
  rnd = np.random.RandomState(0)
  z = rnd.randn(1, *Gs.input_shape[1:])
  tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})
  fakeImg = Gs.run(z, None, **Gs_kwargs)
  print(fakeImg.shape)
  nrChannels = fakeImg.shape[1]
  width = fakeImg.shape[2] # height = width

  return width, nrChannels, fakeImg
     
  
#----------------------------------------------------------------------------

def project_image(proj, targets, png_prefix, num_snapshots):
    snapshot_steps = set(proj.num_steps - np.linspace(0, proj.num_steps, num_snapshots, endpoint=False, dtype=int))
    print('target.shape', targets.shape)
    misc.save_image_grid(targets, png_prefix + 'target.png', drange=[-1,1])
    proj.start(targets)
    while proj.get_cur_step() < proj.num_steps:
        print('\r%d / %d ... ' % (proj.get_cur_step(), proj.num_steps), end='', flush=True)
        proj.step()
        if proj.get_cur_step() in snapshot_steps:
            misc.save_image_grid(proj.get_images(), png_prefix + 'corrupted-step%04d.png' % proj.get_cur_step(), drange=[-1,1])
            cleanImgFile = png_prefix + 'clean-step%04d.png' % proj.get_cur_step()
            print('Saving checkpoint %s' % cleanImgFile )
            misc.save_image_grid(proj.get_clean_images(), cleanImgFile, drange=[-1,1])
    print('\r%-30s\r' % '', end='', flush=True)

#----------------------------------------------------------------------------

def get_img_list(data_dir):
  imgs = []
  valid_images = [".jpg",".gif",".png", '.jpeg']
  img_list = np.sort(os.listdir(data_dir))
  for f in img_list:
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    imgs.append(os.path.join(data_dir,f))

  return imgs

def checkDimensions(fakeImg, inputChannels, inputWidth, inputHeight):
  print('fakeImg.shape f(G(z))', fakeImg.shape)
  
  genChannels = fakeImg.shape[1] # channels of f(G(z)) generated image 
  genWidth = fakeImg.shape[2]
  genHeight = fakeImg.shape[3]
  if inputChannels != genChannels:
    raise ValueError('The generated image is %d-channel, but the input image is %d-channel' % (genChannels, inputChannels))
  if (inputWidth != genWidth) or (inputHeight != genHeight):
    raise ValueError('The generated image has resolution %dx%d, but the input image has resolution %dx%d' % (genWidth, genHeight, inputWidth, inputHeight))



def recon_real_one_img(network_pkl, img, mask_dir, num_snapshots, recontype, superres_factor, num_steps):
    if not runSerial:
      sys.stdout = open('logs/' + str(os.getpid()) + ".out", "w")
      #sys.stderr = open(str(os.getpid()) + ".err", "w")

    
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    imgSize, nrChannelsNet, fakeImg = getImgSize(Gs)
    imgShort = img.split('/')[-1] # without full path
    forward, forwardTrue = constructForwardModel(recontype, imgSize, nrChannelsNet, mask_dir, imgShort, superres_factor)

    print('Loading image "%s"...' % img)
    image = cv2.imread(img, cv2.IMREAD_UNCHANGED) # without this flat, cv2 automatically converts to 3-channels
    ndim = len(image.shape)
    print('Input image has shape:', image.shape)
    if ndim == 2:
      nrChannels = 1
      image = image.reshape(1, image.shape[0], image.shape[1], 1)
    else:
      nrChannels = image.shape[-1]
      if nrChannels == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert from BGR to RGB
      
      image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
      

    #fakeImg = forward(fakeImg) # sample of f(G(z)) for checking dimensions and channels
    checkDimensions(fakeImg, nrChannels, inputWidth=image.shape[1], inputHeight=image.shape[2])

    image = np.transpose(image, (0,3,1,2))# convert from NHWC to NCWH
    print('reshaped shape:', image.shape)
    image = misc.adjust_dynamic_range(image, [0, 255], [-1, 1])
    # save true image
    png_prefix = dnnlib.make_run_dir_path(imgShort[:-4] + '-')
    misc.save_image_grid(image, png_prefix + 'true.png', drange=[-1,1])
    imgTrue = image # this is the true image

    # generate corrupted image, with true forward corruption model
    imgCorrupted = forwardTrue(tf.convert_to_tensor(image))
    imgCorrupted = imgCorrupted.eval()
    proj = projector.Projector(forward, num_steps)
    proj.set_network(Gs)
    project_image(proj, targets=imgCorrupted, png_prefix=png_prefix, num_snapshots=num_snapshots)

    imgRecon = proj.get_clean_images()
    if recontype == 'inpaint':
      # for inpainting, also merge the reconstruction with target image
      imgMerged = tf.where(forward.mask, imgRecon, imgCorrupted).eval() # if true, then recon, else imgCorrupted
      misc.save_image_grid(imgMerged, png_prefix + 'merged.png', drange=[-1,1])
    else:
      imgMerged = None

    return imgCorrupted, imgRecon, imgMerged, imgTrue
      

def recon_real_images(network_pkl, input, masks, num_snapshots, recontype, superres_factor, num_steps):

    img_list = get_img_list(input)
    num_images = len(img_list)
    for image_idx in range(num_images):
      print('Processing image %d/%d' % (image_idx+1, num_images))

      if runSerial:
        recon_real_one_img(network_pkl, img_list[image_idx], masks, num_snapshots, recontype, superres_factor, num_steps)
      else:
        p = multiprocessing.Process(target=recon_real_one_img, args=[network_pkl, dataset_name, img_list[image_idx], masks, num_snapshots, recontype, superres_factor, num_steps])
        p.start()
        p.join()        

#----------------------------------------------------------------------------

def _parse_num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

_examples = '''examples:

  # Recon real images
   python recon.py recon-real-images --input=datasets/brains --masks=masks/256x256 --tag=brains --network=models/brains.pkl --recontype=inpaint


'''

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='''StyleGAN2 projector.

Run 'python %(prog)s <subcommand> --help' for subcommand help.''',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    recon_real_images_parser = subparsers.add_parser('recon-real-images', help='Project real images')
    recon_real_images_parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    recon_real_images_parser.add_argument('--input', help='Directory with input images', required=True)
    recon_real_images_parser.add_argument('--masks', help='Directory with masks (inpainting only). Mask filenames should be identical to the input filenames', required=False)
    recon_real_images_parser.add_argument('--tag', help='Tag for results directory', default='')
    recon_real_images_parser.add_argument('--num-snapshots', type=int, help='Number of snapshots (default: %(default)s)', default=20)
    recon_real_images_parser.add_argument('--num-gpus', type=int, help='Number of gpus (default: %(default)s)', default=1)
    recon_real_images_parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    recon_real_images_parser.add_argument('--recontype', help='Type of reconstruction: "none" (normal image inversion), "super-resolution", "inpaint" (default: %(default)s)', default='none')
    recon_real_images_parser.add_argument('--superres-factor', help='Super-resolution factor: 2,4,8,16,32,64 (default: %(default)s)', type=int, default=4)
    recon_real_images_parser.add_argument('--num-steps', help='Number of iteration steps: <1000 (fast) to 5000 (slow, but better results)  (default: %(default)s)', type=int, default=5000)

    args = parser.parse_args()
    subcmd = args.command
    if subcmd is None:
        print ('Error: missing subcommand.  Re-run with --help for usage.')
        sys.exit(1)

    kwargs = vars(args)
    sc = dnnlib.SubmitConfig()
    sc.num_gpus = args.num_gpus
    kwargs.pop('num_gpus') 
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = kwargs.pop('result_dir')
    print('args:', args)
    command = kwargs.pop('command') 
    sc.run_desc = '%s-%s' %( args.tag, args.recontype)
    tag = kwargs.pop('tag') 

    func_name_map = {
        'recon-real-images': 'recon.recon_real_images'
    }
    dnnlib.submit_run(sc, func_name_map[subcmd], **kwargs)

    

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
