#!/usr/bin/env python
from __future__ import print_function
import os, sys, argparse, shlex, subprocess

verbose = True

#================================#
# ==== User Input Arguments ==== #
#================================#
def get_args(valid_modes):
  parser = argparse.ArgumentParser(description='VIMS Lab UNet Driver v0.10')
  parser.add_argument("--seed", type=int, dest='seed', default=1337)
  parser.add_argument("--mode", type=str, choices=valid_modes, required=True,
                      help="Enter a run mode for the script")
  parser.add_argument("--input-path", type=str, dest='input_path', required=True,
                      help="Path to training or prediction data folder")
  parser.add_argument("--norm-path", type=str, dest='norm_path', required=True,
                      help="Path to normalization parameters file")
  parser.add_argument("--seg-channels", type=str, dest='seg_channels', default="0,1,2",
                      help="Channels of input image to segment. Specified as a commas separated list of channels (Default: 0,1,2 [RGB])")
  parser.add_argument("--image-ext", type=str, dest='input_img_extension', default='.tif',
                      help="Extension of the training images (Default: .tif)")
                      
  # Prediction Specific Params
  parser.add_argument("--weights-path", type=str, dest='weights_path', 
                      help="(Prediction Only) Path to trained network weights")
  parser.add_argument("--output-path", type=str, dest='output_path', 
                      help="(Prediction Only) Path to save segmented images")
  parser.add_argument("--num-classes", type=int, dest='num_classes', default=1,
                      help="(Prediction Only) Number of classes to segment (Default: 1 [Binary Segmentation])")
  parser.add_argument("--output-ext", type=str, dest='output_img_extentsion', default='.tif',
                      help="(Prediction Only) Extension of the output image (Default: .tif)")
                      
  # Training Specific Params 
  parser.add_argument("--checkpoint-directory", type=str, dest='checkpoint_dir', default='checkpoints',
                      help="(Training Only) Directory to save weight checkpoints (Default: ./checkpoints)")
  parser.add_argument("--epochs", type=int, dest='epochs', default=30,
                      help="(Training Only) Number of epochs for training (Default: 30)")
  parser.add_argument("--batch-size", type=int, dest='batch_size', default=32,
                      help="(Training Only) Size of a batch of patches to be run through the network. *Reduce* this if you encounter GPU memory problems. (Default: 32)")
  parser.add_argument("--patch-size", type=int, dest='patch_size', default=128,
                      help="(Training Only) Size of the dimensions of square patches for training. *Increase* this if you encounter GPU memory problems. (Default: 128)")
  parser.add_argument("--patch-thresh", type=float, dest='patch_thresh', default=0.05,
                      help="(Training Only) Patches will only be kept if at least <patch_thresh> percent of the corresponding mask contains true pixels (Default: 0.05)")
  parser.add_argument("--mask-ext", type=str, dest='train_mask_extension', default='.tif',
                      help="(Training Only) Extension of the training masks (Default: .tif)")
  parser.add_argument("--no-shuffle", action='store_true',  dest='no_shuffle', 
                      help="(Training Only) *Don't* shuffle training data")
  return parser.parse_args()

# Parse inference params
valid_modes = ['train', 'predict', 'predict_ts']
args = get_args(valid_modes)
mode = args.mode

# General Params
seed = args.seed
input_path = args.input_path
norm_path = args.norm_path
input_img_extension = args.input_img_extension
seg_channels = args.seg_channels

# Training Params
checkpoint_dir = args.checkpoint_dir
epochs = args.epochs
batch_size = args.batch_size
patch_size = args.patch_size
patch_thresh = args.patch_thresh
train_mask_extension = args.train_mask_extension
no_shuffle = args.no_shuffle

# Inference Params
weights_path = args.weights_path
output_path = args.output_path
output_img_extentsion = args.output_img_extentsion
num_classes = args.num_classes

#==================================#
# ==== Verification of Inputs ==== #
#==================================#

# Script paths
self_path = os.path.realpath(__file__)
self_dir = os.path.dirname(self_path)
scripts_dir = os.path.join(self_dir, 'scripts')
unet_script_path = os.path.join(scripts_dir, 'unet_seg.py') 

# Verify all paths exist before submitting slurm job
if not os.path.isfile(unet_script_path):
  sys.exit("FATAL Error: unet_seg.py could not be found!")
  
if not (os.path.isdir(input_path)):
  sys.exit("Error: input_path directory does not exist!")
  
# If seg_channels doesn't contain only commas and numbers
if any(sc not in [','] + [str(r) for r in range(10)] for sc in seg_channels):
  sys.exit("Error: seg_channels can only contain commas and numbers (no spaces)!")
  
# Verify Predict Parameters
if 'predict' in mode:
  if not os.path.isfile(norm_path):
    sys.exit("Error: norm_path could not be found!")

  if not os.path.isfile(weights_path):
    sys.exit("Error: weights_path could not be found!")  
    
  if not os.path.isdir(output_path):
    print("Warning: output_path directory does not exist; this directory will be created... (%s)" % (output_path,))
    # Verify the containing directory exists for the folder to be made...
    # (i.e. if /a/b/c doesn't exist, verify /a/b exists to make dir c)
    up_one_output_path = os.path.dirname(output_path if output_path[-1] != '/' else output_path[:-1])
    if not os.path.isdir(up_one_output_path):
      sys.exit("Error: The directory containing output_path (%s) doesn't exist either! Exiting..." % (up_one_output_path,))

  if (num_classes < 1 or num_classes > 255):
    sys.exit("Error: num_classes is an invalid value (0 < num_classes < 256)! Exiting...") 
    
# Verify Training Params
if 'train' in mode:
  if not os.path.dirname(norm_path):
    sys.exit("Error: The directory to save norm_path (%s) in cannot be found! Exiting..." % (norm_path,))

  if not os.path.isdir(checkpoint_dir):
    print("Warning: checkpoint_dir directory does not exist; this directory will be created... (%s)" % (checkpoint_dir,))
    # Verify the containing directory exists for the folder to be made...
    # (i.e. if /a/b/c doesn't exist, verify /a/b exists to make dir c)
    up_one_checkpoint_dir = os.path.dirname(checkpoint_dir if checkpoint_dir[-1] != '/' else checkpoint_dir[:-1])
    if not os.path.isdir(up_one_checkpoint_dir):
      sys.exit("Error: The directory containing checkpoint_dir (%s) doesn't exist either! Exiting..." % (up_one_checkpoint_dir,))
  
  if os.path.isfile(checkpoint_dir):
    sys.exit("Error: checkpoint_dir should be a directory not a file! Exiting...")  


#===============================#
# ==== Build Slurm Command ==== #
#===============================#

# Define debug print function if verbose mode is on
debug_print = print if verbose else lambda *args: None

# Initialize Sbatch Command
sbatch_flags = ["-N 1",
                "-c 12",
                "--time=5-12",
                "--mem=32000",
                "--partition=gpu",
                "--account=gpu"]
  
debug_print ("============================")
debug_print ("Sbatch Flags:")
debug_print ("\n".join(sbatch_flags))

# Construct UNet command
# Ex: python -u unet_seg.py --mode <mode> --input-path <input_path> ...
segment_command = ["python", "-u", unet_script_path, 
                    "--seed", str(seed),
                    "--mode", mode,
                    "--input-path", str(input_path),
                    "--norm-path", str(norm_path),
                    "--seg-channels", str(seg_channels),
                    "--image-ext", str(input_img_extension),
                    "--weights-path", str(weights_path), 
                    "--output-path", str(output_path),
                    "--num-classes", str(num_classes),
                    "--output-ext", str(output_img_extentsion),
                    "--checkpoint-directory", str(checkpoint_dir),
                    "--epochs", str(epochs),
                    "--batch-size", str(batch_size),
                    "--patch-size", str(patch_size),
                    "--patch-thresh", str(patch_thresh),
                    "--mask-ext", str(train_mask_extension),
                  ]
                      
segment_command_str = " ".join(segment_command)
debug_print ("============================")
debug_print ("Segment command:")
debug_print (segment_command_str)
segment_command_wrapped = '--wrap="' + segment_command_str + '"'

# Construct sbatch command
sbatch_command = ["sbatch"] + sbatch_flags + [segment_command_wrapped]
sbatch_command_str = " ".join(sbatch_command)
debug_print ("============================")
debug_print ("Full SBatch command:")
debug_print (sbatch_command_str)
debug_print ("============================")

final_sbatch_command = shlex.split(sbatch_command_str)
p = subprocess.Popen(final_sbatch_command)
p.wait()
debug_print ("Success!")
debug_print ("============================")
sys.stdout.flush()
