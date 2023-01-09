# VIMSStromuleSegmentation

A Python package containing two command scripts (entry points) for:

- `unet_seg`: training a stromule segmentation neural network
- `UnetSegmentation`: submission of a Slurm job running the `unet_seg` command

The package can be installed from the directory containg this README.md file using the command:

```bash
$ python3 -m pip install <source directory>
```

The package requires the scikit-image, keras, tensorflow, and numpy packages.


## Installation in a dedicated virtual environment

The dependencies of this package make it convenient for a Python virtual environment to be used.  Using the Intel oneAPI Anaconda support, we start by creating the target virtualenv:

```bash
$ conda create --prefix="$(pwd)/venv" --channel=intel numpy keras scikit-image tensorflow
Collecting package metadata (current_repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: /home/1001/VIMSStromuleSegmentation/venv

  added / updated specs:
    - keras
    - numpy
    - scikit-image
    - tensorflow
       :
Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use
#
#     $ conda activate /home/1001/VIMSStromuleSegmentation/venv
#
# To deactivate an active environment, use
#
#     $ conda deactivate
```

After activating the virtualenv, the VIMSStromuleSegmentation package can be installed:

```bash
$ conda activate /home/1001/VIMSStromuleSegmentation/venv
$ python3 -m pip install "$(pwd)"
Processing /home/1001/VIMSStromuleSegmentation
  Preparing metadata (setup.py) ... done
    :
Successfully installed VIMSStromuleSegmentation-1.0
```

This makes the `UnetSegmentation` and `unet_seg` commands available:

```bash
$ which unet_seg
~/VIMSStromuleSegmentation/venv/bin/unet_seg

$ unet_seg --help
usage: unet_seg [-h] [--seed SEED] -m {train,predict,predict_ts} -i INPUT_PATH -n NORM_PATH [-c SEG_CHANNELS]
                [-e INPUT_IMG_EXTENSION] [--weights-path WEIGHTS_PATH] [--output-path OUTPUT_PATH]
                [--num-classes NUM_CLASSES] [--output-ext OUTPUT_IMG_EXTENTSION]
                [--checkpoint-directory CHECKPOINT_DIR] [--epochs EPOCHS] [--batch-size BATCH_SIZE]
                [--patch-size PATCH_SIZE] [--patch-thresh PATCH_THRESH] [--mask-ext TRAIN_MASK_EXTENSION]
                [--no-shuffle]

Unet Driver

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED
  -m {train,predict,predict_ts}, --mode {train,predict,predict_ts}
                        Enter a run mode for the script
  -i INPUT_PATH, --input-path INPUT_PATH
                        Path to training or prediction data folder
  -n NORM_PATH, --norm-path NORM_PATH
                        Path to normalization parameters file
  -c SEG_CHANNELS, --seg-channels SEG_CHANNELS
                        Channels of input image to segment. Specified as a commas separated list of channels (Default:
                        0,1,2 [RGB])
  -e INPUT_IMG_EXTENSION, --image-ext INPUT_IMG_EXTENSION
                        Extension of the training images (Default: .tif)
  --weights-path WEIGHTS_PATH
                        Path to trained network weights
  --output-path OUTPUT_PATH
                        Path to save segmented images
  --num-classes NUM_CLASSES
                        Number of classes to segment (Default: 1 [Binary Segmentation])
  --output-ext OUTPUT_IMG_EXTENTSION
                        Extension of the output image (Default: .tif)
  --checkpoint-directory CHECKPOINT_DIR
                        Directory to save weight checkpoints (Default: checkpoints)
  --epochs EPOCHS       Number of epochs for training (Default: 30)
  --batch-size BATCH_SIZE
                        Size of a batch of patches to be run through the network. *Reduce* this if you encounter GPU
                        memory problems. (Default: 32)
  --patch-size PATCH_SIZE
                        Size of the dimensions of square patches for training. *Increase* this if you encounter GPU
                        memory problems. (Default: 128)
  --patch-thresh PATCH_THRESH
                        Patches will only be kept if at least <patch_thresh> percent of the corresponding mask
                        contains true pixels (Default: 0.05)
  --mask-ext TRAIN_MASK_EXTENSION
                        Extension of the training masks (Default: .tif)
  --no-shuffle          Don't shuffle training data.
```
