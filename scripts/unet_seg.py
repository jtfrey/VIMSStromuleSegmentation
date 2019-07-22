from __future__ import print_function
import argparse
import numpy as np

########################
# Arguments and Params #
########################
def get_args(valid_modes):
  parser = argparse.ArgumentParser(description='Unet Driver')
  ###
  parser.add_argument("--seed", type=int, dest='seed', default=1337)
  ###
  parser.add_argument("-m", "--mode", type=str, choices=valid_modes, required=True,
                      help="Enter a run mode for the script")
  ###
  parser.add_argument("-i", "--input-path", type=str, dest='input_path', required=True,
                      help="Path to training or prediction data folder")
  ###
  parser.add_argument("-n", "--norm-path", type=str, dest='norm_path', required=True,
                      help="Path to normalization parameters file")
  ###
  parser.add_argument("-c", "--seg-channels", type=str, dest='seg_channels', default="0,1,2",
                      help="Channels of input image to segment. Specified as a commas separated list of channels (Default: 0,1,2 [RGB])")
  ###
  parser.add_argument("-e", "--image-ext", type=str, dest='input_img_extension', default='.tif',
                      help="Extension of the training images (Default: .tif)")
  
  # Inference Specific Params
  ###
  parser.add_argument("--weights-path", type=str, dest='weights_path', 
                      help="Path to trained network weights")
  ###
  parser.add_argument("--output-path", type=str, dest='output_path', 
                      help="Path to save segmented images")
  ###
  parser.add_argument("--num-classes", type=int, dest='num_classes', default=1,
                      help="Number of classes to segment (Default: 1 [Binary Segmentation])")
  ###
  parser.add_argument("--output-ext", type=str, dest='output_img_extentsion', default='.tif',
                      help="Extension of the output image (Default: .tif)")
  
  # Training Specific Params
  ### 
  parser.add_argument("--checkpoint-directory", type=str, dest='checkpoint_dir', default='checkpoints',
                      help="Directory to save weight checkpoints (Default: checkpoints)")
  ###
  parser.add_argument("--epochs", type=int, dest='epochs', default=30,
                      help="Number of epochs for training (Default: 30)")
  ###
  parser.add_argument("--batch-size", type=int, dest='batch_size', default=32,
                      help="Size of a batch of patches to be run through the network. *Reduce* this if you encounter GPU memory problems. (Default: 32)")
  ###
  parser.add_argument("--patch-size", type=int, dest='patch_size', default=128,
                      help="Size of the dimensions of square patches for training. *Increase* this if you encounter GPU memory problems. (Default: 128)")
  ###
  parser.add_argument("--patch-thresh", type=float, dest='patch_thresh', default=0.05,
                      help="Patches will only be kept if at least <patch_thresh> percent of the corresponding mask contains true pixels (Default: 0.05)")
  ###
  parser.add_argument("--mask-ext", type=str, dest='train_mask_extension', default='.tif',
                      help="Extension of the training masks (Default: .tif)")
  ###
  parser.add_argument("--no-shuffle", action='store_true',  dest='no_shuffle', 
                      help="Don't shuffle training data.")
  ###
  return parser.parse_args()

# Parse inference params
valid_modes = ['train', 'predict', 'predict_ts']
args = get_args(valid_modes)
mode = args.mode

# Random Seed
seed = args.seed
np.random.seed(seed)

# General Params
input_img_extension = args.input_img_extension
input_path = args.input_path
norm_path = args.norm_path
segmentation_channels = [int(ch) for ch in args.seg_channels.split(',')]

# Training Params
checkpoint_dir = args.checkpoint_dir
epochs = args.epochs
batch_size = args.batch_size
patch_size = args.patch_size
patch_thresh = args.patch_thresh
train_mask_extension = args.train_mask_extension
shuffle = not args.no_shuffle

# Inference Params
weights_path = args.weights_path
output_path = args.output_path
output_image_extension = args.output_img_extentsion
model_output_channels = args.num_classes
print ("====================")

#############################
# Import Required Libraries #
#############################
import os, sys
from datetime import datetime
from skimage import io
from utils import *
  
####################
# Model Parameters #
####################
MODEL = big_unet
LOSS = tversky_loss

############
# Training #
############
if mode == 'train':

  # Load Training Images
  patch_shape = (patch_size, patch_size)
  print ("====================")
  print ("Loading Training Patches...")
  train_images, train_masks = get_patches(input_path, 
                                          segmentation_channels, 
                                          patch_shape, 
                                          input_img_extension, 
                                          train_mask_extension, 
                                          patch_thresh=patch_thresh)
  if len(train_images.shape) < 4:
    train_images = np.expand_dims(train_images, axis=-1)
  
  print ("All Patches Loaded!")
  print (" >> Number of Training Patches:", train_images.shape[0], "| Patch Shape:", train_images.shape[1:])
  print (" >> Number of Mask Patches:", train_masks.shape[0], "| Patch Shape:", train_masks.shape[1:])
  
  # Shuffle Images
  if shuffle:
    print ("====================")
    print ("Shuffling image and mask patches...")
    n_s = np.arange(train_images.shape[0])
    np.random.shuffle(n_s)
    train_images = train_images[n_s]
    train_masks = train_masks[n_s]
    print ("Patches shuffled!")
  
  # Calculate Normalization Parameters
  print ("====================")
  print ("Calculating normalization parameters and saving...")
  means = []
  stds = []
  for class_i in range(train_images.shape[-1]):
    means.append( np.mean(train_images[...,class_i]) )
    stds.append( np.std(train_images[...,class_i]) )
    train_images[...,class_i] -= means[class_i]
    train_images[...,class_i] /= stds[class_i]
  
  # Save Normalization Parameters
  with open(norm_path, 'w') as fp:
    for class_i in range(train_images.shape[-1]):
      fp.write(str(means[class_i]) + ',' + str(stds[class_i]) + '\n')
  print ("Normalization parameters saved!")
  print (" >> Path:", norm_path)
  print (" >> Means:", str(means), "| Stdevs:", str(stds))
  
  # Determine the number of input and output channels
  in_ch = int(train_images.shape[-1])
  out_ch= int(train_masks.shape[-1])
  
  # Setup Checkpoint Path
  print ("====================")
  print ("Setting up checkpoint callback and checkpoint directory...")
  if not os.path.isdir(checkpoint_dir):
    print (" >> checkpoint_directory doesn't exist; attempting to create it...")
    try:
      os.makedirs(checkpoint_dir)
      print ("   >> Created checkpoint_directory:", checkpoint_dir)
    except:
      sys.exit("Invalid checkpoint_dir path! Exiting...")
  date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
  weights_filename = date_str + '_E{epoch:02d}_L{val_loss:.4f}_A{val_acc:.3f}'
  weights_filename += '_in' + str(in_ch) + '_out' + str(out_ch) + '.h5'
  checkpoint_path = os.path.join(checkpoint_dir, weights_filename)
  print (" >> Checkpoint filename format:", weights_filename)
  
  print ("====================")
  print ("Creating model and training...")
  print (" >> Model input channels:", in_ch, "| Model output channels:", out_ch)
  print (" >> Batch shape:", (batch_size,) + train_images.shape[1:] )
  # Define Model and Model Checkpoint Callback
  model = MODEL(patch_size, patch_size, in_ch=in_ch, out_ch=out_ch, loss_fn=LOSS)
  ckpt = ModelCheckpoint(str(checkpoint_path), verbose=2, monitor='val_loss', mode='auto', save_best_only=True)
  print ("Model created!")
  print ("====================")
  
  # Train the Model
  model.fit(train_images, train_masks, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True, validation_split=0.15, callbacks=[ckpt])
  print ("====================")
  print ("Training Completed!")

#############
# Inference #
#############
elif mode == 'predict' or mode == 'predict_ts':

  # Gather inference image filenames
  print ("====================")
  print ("Gathering input image paths...")
  inference_image_files = get_paths(input_path, input_img_extension) #[if for f in os.listdir(input_path) if f.endswith(input_img_extension)]
  num_image_files = len(inference_image_files)
  print (" >> Total number of images:", num_image_files)
  print ("Paths gathered!")
  
  # If time series, set the time_series flag
  time_series = mode == 'predict_ts'
  
  print ("====================")
  print ("Setting up output path...")
  if os.path.isdir(output_path):
    print (" >> Output directory exists; this may overwrite data!")
  else:
    print (" >> Output directory does not exist; attempting to create it...")
    try:
      os.makedirs(output_path)
      print (" >> Output directory created!")
    except:
      pass
  print ("Output path set up!")
  
  # Load one image and compute shape to initialize network
  print ("====================")
  print ("Calculating image size and pad amount...")
  mult16 = [16*i for i in range(128)]
  init_img = io.imread(inference_image_files[0]).astype(np.float32)
  if len(init_img.shape) < 3:
    init_img = np.expand_dims(init_img, axis=-1)
  original_image_shape = init_img.shape if not time_series else init_img.shape[1:]
  print (" >> Initial image xy shape:", original_image_shape[:2])
  if original_image_shape[0] not in mult16:
    y_ind = next(x[0] for x in enumerate(mult16) if x[1] > original_image_shape[0])
    new_y_sz = mult16[y_ind]
    pad_y = new_y_sz - original_image_shape[0]
  else:
    new_y_sz = original_image_shape[0]
    pad_y = 0
  if original_image_shape[1] not in mult16:
    x_ind = next(x[0] for x in enumerate(mult16) if x[1] > original_image_shape[1])
    new_x_sz = mult16[x_ind]
    pad_x = new_x_sz - original_image_shape[1]
  else:
    new_x_sz = original_image_shape[1]
    pad_x = 0
  if pad_x > 0 or pad_y > 0:
    print (" >> New image xy shape:", (new_y_sz, new_x_sz))
  print ("Image size and padding calculated and image padded!")
  
  # Load normalization parameters
  print ("====================")
  print ("Loading normalization parameters...")
  means = []
  stds = []
  with open(norm_path, 'r') as fp:
    for line in fp.readlines():
      tmp_mean, tmp_std = line.strip().split(',')
      means.append(float(tmp_mean))
      stds.append(float(tmp_std))
  if len(means) != len(segmentation_channels):
    print (" >> Number of means/stds:", len(means), "| Number of segmentation channels:", len(segmentation_channels))
    sys.exit("Error: Number of normalization parameters do not match the number of segmentation channels... Exiting...")
  print ("Normalization parameters loaded!")
  
  # Create model and load weights
  print ("====================")
  print ("Creating model and loading weights...")
  print (" >> Model input shape:", (new_y_sz, new_x_sz))
  print (" >> Model input channels:", len(segmentation_channels), "| Model output channels:", model_output_channels)
  model = MODEL(new_y_sz, new_x_sz, in_ch=len(segmentation_channels), out_ch=model_output_channels)
  model.load_weights(weights_path)
  print ("Model created!")

  def infer_on_image(img, output_name, time_point=None):
    print ("  ---")
    print ("  Splitting input image into only segmentation channels and normalizing...")
    # Give image a single channel dimension if needed
    if len(img.shape) < 3:
      print ("   >> No channel dimension, expanding image to have 1 channel...")
      img = np.expand_dims(img, axis=-1)
        
    # Split by segmentation_channels and normalize
    print ("   >> Segmentation channels:", segmentation_channels)
    print ("   >> Input image channel dimension:", img.shape[-1])
    n_ch = len(segmentation_channels)
    if n_ch < img.shape[-1]:
      new_img = np.zeros((img.shape[:-1] + (n_ch,)), dtype=np.float32)
      for ch_i in range(n_ch):
        new_img[...,ch_i] = img[...,segmentation_channels[ch_i]]
      img = new_img
    else:
      print ("   >> Error: Too many segmentation_channels given! Ignoring segmentation_channels and using the entire input image...")
    
    img = img / np.max(img)
    for class_i in range(img.shape[-1]):
      img[...,class_i] -= means[class_i]
      img[...,class_i] /= stds[class_i]
    print ("  Image split and normalized!")
    
    # Pad image size to be multiple of 16 for inference
    if pad_x > 0 or pad_y > 0:
      print ("  ---")
      print ("  Padding image so that its xy dimensions are multiples of 16...")
      print ("   >> Initial image shape:", img.shape)
      img = np.pad(img, ((0, pad_y), (0, pad_x), (0, 0)), 'constant')
      print ("   >> Padded image shape:", img.shape)
    
    # Give image a batch dimension
    img = np.expand_dims(img, axis=0)
    
    # Predict the segmentation of the image and save
    print ("  ---")
    print ("  Predicting segmentation of image...")
    tmp_seg_img = model.predict(img)
    print ("  Prediction complete!")
  
    # Resize output image to the correct shape
    print ("  ---")
    print ("  Resizing output segmentation...")
    seg_img = tmp_seg_img.reshape((img.shape[1], img.shape[2], model_output_channels))
    
    # If image isn't 3 channel at least, make it 3 channels
    if model_output_channels < 3:
      print ("   >> Making output image at *least* 3-channel for easier viewing...")
      new_img = np.zeros((img.shape[1], img.shape[2],3))
      new_img[:,:,:model_output_channels] = seg_img
      seg_img = new_img
    print ("  Resizing done!")
    
    # Set output path and save (save image as float 16)
    print ("  ---")
    print ("  Saving image...")
    if time_series:
      print ("   >> Saving time point image as frames in directory...")
      ts_output_dir = os.path.join(output_path, output_name)
      print ("   >> Directory of frame images:", ts_output_dir)
      if not os.path.isdir(ts_output_dir):
        print ("   >> Directory doesn't exist... creating!")
        try:
          os.makedirs(ts_output_dir)
        except:
          pass
      elif os.path.isdir(ts_output_dir) and time_point == 0:
        print ("   >> Directory exists on first time point... this may overwrite files!")
      output_file_name = "t" + str(time_point) + "_segmented" + output_image_extension
      output_file_path = os.path.join(ts_output_dir, output_file_name)
    else:
      output_file_name = output_name + "_segmented" + output_image_extension
      output_file_path = os.path.join(output_path, output_file_name)
    
    # Reshape image to original size
    print ("  ---")
    print ("  Resizing image to original unpadded shape...")
    seg_img_unpadded = seg_img[:original_image_shape[0],:original_image_shape[1]]
    print ("   >> Padded image shape:", seg_img.shape, "| New shape:", seg_img_unpadded.shape)
    io.imsave(output_file_path, seg_img_unpadded.astype(np.float16)) 
    print ("  Image saved to:", output_file_path)

  # Loop over all images...
  for image_i, inference_image_path in enumerate(inference_image_files):
  
    # Get inference image path, load image, and normalize
    print ("====================")
    print ("Working on image:", inference_image_path)
    print (" >> Image ", image_i+1, "/", num_image_files)
    img = io.imread(inference_image_path).astype(np.float32)
    
    output_name = inference_image_path.replace(" ","_").split("/")[-1]
    output_name = output_name[:-len(input_img_extension)]
    print (output_name)
    if time_series:
      for time_point in range(img.shape[0]):
        infer_on_image(img[time_point], output_name, time_point=time_point)
    else:
      infer_on_image(img, output_name)
