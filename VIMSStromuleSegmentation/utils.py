import numpy as np
import os, sys, fnmatch
from skimage import io
from skimage.transform import rotate
from skimage.util.shape import view_as_windows
from keras.models import Model
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras.layers import Convolution2D, MaxPooling2D, Lambda, Reshape, Flatten
from keras.layers import Input, Concatenate, UpSampling2D, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.optimizers import Adam

##################
# Path functions #
##################

# Collect paths of images recursively
def get_paths(path, img_extension, subfolder=None):
  if subfolder:
    path = os.path.join(path, subfolder)
    
  img_paths = []
  for root, directories, filenames in os.walk(path):
    for filename in fnmatch.filter(filenames, "*" + img_extension):
      img_paths.append(os.path.join(root,filename))
  
  return img_paths

# Collects image and mask paths for training
def training_paths(path, img_extension, mask_extension):
  img_paths = get_paths(path, img_extension, subfolder='images')
  mask_paths = get_paths(path, mask_extension, subfolder='masks')
  img_paths.sort()
  mask_paths.sort()
  return img_paths, mask_paths
  
# Loads images and makes training patches
def get_patches(train_data_path, segmentation_channels, patch_shape, img_extension, mask_extension, patch_thresh=0.05):
  # Gather paths
  img_paths, mask_paths = training_paths(train_data_path, img_extension, mask_extension)
  if len(img_paths) != len(mask_paths):
    sys.exit("Number of images and masks is inconsistent. Exiting... (# images: %d | # masks: %d)".format(len(img_paths), len(mask_paths)))
  
  train_images = []
  train_masks = []
  n_images = len(img_paths)
  for ind in range(n_images):
    if (ind+1) % 10 == 0 or ind == 0:
      print (" >> Generating patches for image", ind+1, "of", str(n_images) + "...")
      print ("    >> Currently at", len(train_images), "total image pairs")
      
    # Read image, split only the segmentation_channels, and normalize
    img = io.imread(img_paths[ind]).astype(np.float32)
    if len(img.shape) < 3:
      img = np.expand_dims(img, axis=-1)
    n_ch = len(segmentation_channels)
    new_img = np.zeros((img.shape[:-1] + (n_ch,)), dtype=np.float32)
    for ch_i in range(n_ch):
      new_img[...,ch_i] = img[...,segmentation_channels[ch_i]]
    img = new_img
    img = img / np.max(img)
    img_patch_shape = patch_shape + (img.shape[-1],)
    
    # Load masks and normalize
    mask = io.imread(mask_paths[ind]).astype(np.float32)
    mask = mask / np.max(mask)
    if len(mask.shape) < 3:
      mask = np.expand_dims(mask, axis=-1)
    mask_patch_shape = patch_shape + (mask.shape[-1],)
    
    # Create windows from both the training images and the masks
    img_patches = view_as_windows(img, img_patch_shape, step=patch_shape[0]//4)
    mask_patches = view_as_windows(mask, mask_patch_shape, step=patch_shape[0]//4)
    img_patches = np.squeeze(img_patches)
    mask_patches = np.squeeze(mask_patches)
    
    # Set the threshold number of pixels required to add the training image to the set
    # If at least 5% of the mask patch has true values 
    mask_thresh = patch_thresh * patch_shape[0] * patch_shape[1]

    # Iterate over the grid of window views
    for patch_ind_0 in range(img_patches.shape[0]):
      for patch_ind_1 in range(img_patches.shape[1]):
        img_patch = np.array(img_patches[patch_ind_0,patch_ind_1])
        mask_patch = np.array(mask_patches[patch_ind_0,patch_ind_1])
        
        # Only include training images where the mask patch is above the threshold
        if (mask_patch > 0).sum() > mask_thresh:
          # Expand dims to make patch "1 channel"
          # img_patch = np.expand_dims(img_patch, axis=-1)

          # Append Patches to list
          train_images.append(img_patch)
          train_masks.append(mask_patch)

          # Augment patches and append those as well
          img90 = np.rot90(img_patch)
          mask90 = np.rot90(mask_patch)
          train_images.append(img90)
          train_masks.append(mask90)

          img180 = np.rot90(img90)
          mask180 = np.rot90(mask90)
          train_images.append(img180)
          train_masks.append(mask180)

          img270 = np.rot90(img180)
          mask270 = np.rot90(mask180)
          train_images.append(img270)
          train_masks.append(mask270)

          train_images.append(np.fliplr(img_patch))
          train_masks.append(np.fliplr(mask_patch))
          train_images.append(np.flipud(img_patch))
          train_masks.append(np.flipud(mask_patch))

  return np.array(train_images, dtype=np.float32), np.array(train_masks, dtype=np.float32)
  
##########
# Losses #
##########

smooth = 1e-5
def dice_coef(y_true, y_pred):
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  intersection = K.sum(y_true_f * y_pred_f)
  return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
  
def dice_loss(y_true, y_pred):
  return -dice_coef(y_true, y_pred)

def tversky_loss(y_true, y_pred):
    alpha = 0.5
    beta  = 0.5
    ones = K.ones(K.shape(y_true))
    p0 = y_pred      # proba that voxels are class i
    p1 = ones-y_pred # proba that voxels are not class i
    g0 = y_true
    g1 = ones-y_true
    
    num = K.sum(p0*g0, (0,1,2))
    den = num + alpha*K.sum(p0*g1,(0,1,2)) + beta*K.sum(p1*g0,(0,1,2))
    T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]
    
    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl-T

def multi_dice_coef_loss(num_classes, channel_weights=None):
  def _multi_dice_coef_loss(y_true, y_pred):
    loss = 0.
    total = K.sum(K.flatten(y_true)) + smooth
    denominator = 0.
    for i in range(num_classes):
      denominator += total / (K.sum(K.flatten(y_true[...,i])) + smooth)
    
    for i in range(num_classes):
      ratio_i = total / K.sum(K.flatten(y_true[...,i]))
      ratio_i = ratio_i / denominator
      if channel_weights:
        ratio_i = ratio_i * channel_weights[i]
      loss += ratio_i * dice_loss(y_true, y_pred)
    return loss
  return _multi_dice_coef_loss
  
#####################
# Unet Architecture #
#####################

def unet(patch_height, patch_width, in_ch, out_ch, loss_fn=None):
  inputs = Input((patch_height, patch_width, in_ch))
  conv1 = Convolution2D(32, (3, 3), activation='relu', padding='same')(inputs)
  conv1 = BatchNormalization()(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

  conv2 = Convolution2D(64, (3, 3), activation='relu', padding='same')(pool1)
  conv2 = BatchNormalization()(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

  conv3 = Convolution2D(128, (3, 3), activation='relu', padding='same')(pool2)
  conv3 = BatchNormalization()(conv3)

  up1 = Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(conv3), conv2])
  conv4 = Convolution2D(64, (3, 3), activation='relu', padding='same')(up1)
  conv4 = BatchNormalization()(conv4)

  up2 = Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(conv4), conv1])
  conv5 = Convolution2D(32, (3, 3), activation='relu', padding='same')(up2)
  conv5 = BatchNormalization()(conv5)

  conv7 = Convolution2D(16, (1, 1), activation='relu')(conv5)
  conv7 = BatchNormalization()(conv7)
  conv7 = Dropout(rate=0.33)(conv7)
  last_conv = Convolution2D(out_ch, (1, 1), activation='sigmoid')(conv7)
  
  model = Model(inputs=inputs, outputs=last_conv)
  if loss_fn:
    model.compile(optimizer=Adam(lr=1e-4), loss=loss_fn, metrics=['accuracy'])
  return model
  
def big_unet(patch_height, patch_width, in_ch, out_ch, loss_fn=None):
  inputs = Input((patch_height, patch_width, in_ch))
  conv1 = Convolution2D(64, (3, 3), activation='relu', padding='same')(inputs)
  conv1 = BatchNormalization()(conv1)
  conv1 = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

  conv2 = Convolution2D(128, (3, 3), activation='relu', padding='same')(pool1)
  conv2 = BatchNormalization()(conv2)
  conv2 = Convolution2D(128, (3, 3), activation='relu', padding='same')(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

  conv3 = Convolution2D(256, (3, 3), activation='relu', padding='same')(pool2)
  conv3 = BatchNormalization()(conv3)
  conv3 = Convolution2D(256, (3, 3), activation='relu', padding='same')(conv3)
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
  
  conv3a = Convolution2D(512, (3, 3), activation='relu', padding='same')(pool3)
  conv3a = BatchNormalization()(conv3a)
  conv3a = Convolution2D(512, (3, 3), activation='relu', padding='same')(conv3a)
  pool3a = MaxPooling2D(pool_size=(2, 2))(conv3a)
  
  conv4 = Convolution2D(512, (3, 3), activation='relu', padding='same')(pool3a)
  conv4 = BatchNormalization()(conv4)
  conv4 = Convolution2D(512, (3, 3), activation='relu', padding='same')(conv4)
  
  up1a = Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(conv4), conv3a])
  conv5a = Convolution2D(512, (3, 3), activation='relu', padding='same')(up1a)
  conv5a = BatchNormalization()(conv5a)
  conv5a = Convolution2D(512, (3, 3), activation='relu', padding='same')(conv5a)
  
  up1 = Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(conv5a), conv3])
  conv5 = Convolution2D(256, (3, 3), activation='relu', padding='same')(up1)
  conv5 = BatchNormalization()(conv5)
  conv5 = Convolution2D(256, (3, 3), activation='relu', padding='same')(conv5)
  
  up1 = Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(conv5), conv2])
  conv6 = Convolution2D(128, (3, 3), activation='relu', padding='same')(up1)
  conv6 = BatchNormalization()(conv6)
  conv6 = Convolution2D(128, (3, 3), activation='relu', padding='same')(conv6)

  up2 = Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(conv6), conv1])
  conv7 = Convolution2D(64, (3, 3), activation='relu', padding='same')(up2)
  conv7 = BatchNormalization()(conv7)
  conv7 = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv7)
  
  conv8 = Convolution2D(32, (1, 1), activation='relu')(conv7)
  conv8 = BatchNormalization()(conv8)
  conv8 = Dropout(rate=0.33)(conv8)
  last_conv = Convolution2D(out_ch, (1, 1), activation='sigmoid')(conv8)

  model = Model(inputs=inputs, outputs=last_conv)
  if loss_fn:
    model.compile(optimizer=Adam(lr=1e-4), loss=loss_fn, metrics=['accuracy'])
  return model
