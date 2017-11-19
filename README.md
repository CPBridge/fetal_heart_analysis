# Fetal Heart Tools

This repository contains all the code to run the experiments in my DPhil thesis "Automated Analysis of Fetal Cardiac Ultrasound Videos", except a few self-contained parts that are stored in my other repositories (also on Github). It implements a tool for automatically estimating information
about each frame in ultrasound videos of the fetal heart, along with the surrounding
experimental workflow. The rest of this readme assumes some familiarity with the work (please refer to the references below).

There are two parts to this repository:

- The `cpp` folder contains C++ code for the image analysis pipeline.
- The `scripts` folder contains a set of supporting Python scripts that automate
the training and testing of large experiments and provide other parts of the
workflow such as analysis of results.

## Quick Install on Ubuntu

The following script handles the entire install process including installing all dependencies, cloning all relevant repositories from Github, and compiling the C++ parts of the project for Ubuntu (it will likely work on most Ubuntu derivatives also). Make sure you are in the directory that you wish to contain the code **before** running it:

```bash
# Install build dependencies
sudo apt install build-essential git cmake

# Install third party C++ libraries
sudo apt install libopencv-dev libboost-all-dev libeigen3-dev

# Install python dependencies
sudo apt install python-numpy python-scipy python-matplotlib python-opencv

# Clone all the necessary repos
git clone https://github.com/CPBridge/RIFeatures.git
git clone https://github.com/CPBridge/monogenic.git
git clone https://github.com/CPBridge/canopy.git
git clone https://github.com/CPBridge/fetal_heart_analysis.git

# Make a build directory
mkdir build
cd build

# Run the build process
cmake -D CANOPY_DIR=../canopy -D RIFEATURES_DIR=../RIFeatures -D MONOGENIC_DIR=../monogenic ../fetal_heart_analysis/cpp/
make
```
For other operating systems, or to have finer control over the installation process, read the full installation instructions below.

## Full Installation Guide

### C++ Code

#### Portability

I use exclusively GNU/Linux-based operating systems for my work and therefore the entire pipeline has been thoroughly tested on Ubuntu and Manjaro (Arch). However, I have been careful to use cross-platform libraries everywhere, along with a cross-platform build tool (CMake) to ensure that there is no reason that the C++ code won't compile and run on Windows and MacOS. That said, I have not been able to test this. If you are using this code on other platforms please let me know how it works out!

#### Dependencies

The following third party open-source C++ libraries are necessary to compile and
run the code in this project. They are all common and available on most platforms
(GNU/Linux, MacOS, and Windows):

* A C++ compiler supporting the C++14 standard (requires a relatively modern version of your compiler).
* A C++ compiler supporting the [OpenMP](http://openmp.org/wp/) standard (includes most major compilers on major platforms including MSVC, g++ and clang).
* The [OpenCV](http://opencv.org) library for computer vision. Tested on version 3.1.0 but most fairly recent
versions should be compatible. If you are using GNU/Linux, there will probably
be a suitable packaged version in your distribution's repository.
* The [boost](http://www.boost.org) [special functions](http://www.boost.org/doc/libs/1_62_0/libs/math/doc/html/special.html), [program options](http://www.boost.org/doc/libs/1_64_0/doc/html/program_options.html), [filesystem](http://www.boost.org/doc/libs/1_62_0/libs/filesystem/doc/index.htm)
and [system](http://www.boost.org/doc/libs/1_62_0/libs/system/doc/index.html) libraries. Again there are likely to be a suitable packaged versions on your GNU/Linux distribution. Typically it is straightforward to install the entire set of boost libraries together.
- The [Eigen](http://eigen.tuxfamily.org) library for linear algebra. Again there are packaged versions for
most package managers.

For example, if you are using Ubuntu or an Ubuntu-based GNU/Linux distribution you can get all the necessary third-party dependencies as follows (tested on Ubuntu 16.04):

```bash
$ sudo apt install build-essential libopencv-dev libboost-all-dev libeigen3-dev

```

Furthermore, you will need a copy of the following code found in my other repositories:

- [CPBridge/RIFeatures](https://github.com/CPBridge/RIFeatures) : An OpenCV/C++ implementation of rotation invariant feature extraction.
- [CPBridge/monogenic](https://github.com/CPBridge/monogenic) : An OpenCV/C++ implementation of the monogenic signal.
- [CPBridge/canopy](https://github.com/CPBridge/canopy) : A header-only C++ library for random forests.

You can straightforwardly clone these from Github with commands like the following on Unix-like systems:

```bash
$ cd /path/to/a/suitable/location
$ git clone https://github.com/CPBridge/RIFeatures.git
$ git clone https://github.com/CPBridge/monogenic.git
$ git clone https://github.com/CPBridge/canopy.git
```

You **do not** need to build or install the code in the above repositories - everything is handled
by the build process for this repository. You just need to download the source code.

#### Compiling

You should be able to use [CMake](https://cmake.org/) to configure your platform's
build process. Perform an out-of-source build and tell CMake to find the source in this repository's
`cpp` directory. You will also need to set some variables within CMake as follows:

- `RIFEATURES_DIR` : the location of the root of the RIFeatures repository on your system.
- `MONOGENIC_DIR` : the location of the root of the monogenic repository on your system.
- `CANOPY_DIR` : the location of the root of the canopy repository on your system.

You can either do this within your preferred CMake GUI (and then hit configure and generate) or pass them in on the command line when you invoke CMake, e.g. if you put all the repositories in your home directory `/home/fred`, then you could use

```bash
$ cmake -D CANOPY_DIR=/home/fred/canopy -D RIFEATURES_DIR=/home/fred/RIFeatures -D MONOGENIC_DIR=/home/fred/monogenic /home/fred/fetal_heart_analysis/cpp/
```

Once CMake has generated the relevant set of build files, you will need to use
your platform's tool to complete the build process. E.g. on GNU/Linux with GNU Make all you need is:

```bash
$ make
```

#### Optimising for Best Performance

In order to achieve high frame rate video processing, as used in my thesis and publications, you will need to perform some basic tuning of the compilation process. The best settings to use will vary between different platforms. In general, you will want to ensure that you are compiling in **release** mode (in CMake set the `CMAKE_BUILD_TYPE` variable to `Release`) and turn on all compiler optimisations.

Wherever I have reported experimental results, I used the following set of compiler flags with GCC on Linux:

`CMAKE_CXX_FLAGS_RELEASE` = `-O3 -DNDEBUG -march=native -DEIGEN_NO_DEBUG -Ofast`

These settings ignore all debug operations (`-DNDEBUG`, and specifically for the Eigen library `-DEIGEN_NO_DEBUG`), make use of any special instructions your processor has available (`-march=native`), and trade off accuracy in favour of speed in mathematical operations (`-Ofast`).

### Scripts

To run the scripts you will need a Python 2 interpreter (I've been using 2.7 but it may work on earlier versions). You will also need the following very common third-party packages, which are available in all good package management systems:

- [numpy](www.numpy.org) : Basic package for numeric/scientific computing
- [scipy](https://www.scipy.org/) : Further numeric and scientific libraries
- [matplotlib](http://matplotlib.org/) : Package for plotting
- OpenCV : you will need to ensure that your installation of OpenCV (see above) is set up to include the python bindings for your interpreter. Often this will have been handled automatically by however you installed OpenCV.

Additionally, you need to make sure that the interpreter can find the `scripts/utilities` directory, as some scripts depend on the modules in there. You can do this by, for example, adding that directory to your `PYTHONPATH` environment variable either temporarily or permanently.

## Overview of Repository Layout

At the top level, the repository is divided into the `cpp` directory, which contains the C++ code for training random forest models and the testing executables, and the `scripts` directory, which contains python scripts to perform various peripheral tasks.

Within the `cpp` directory, the C++ code is organised into the following subdirectories:

- `exec` : This contains the top-level code for four executables. There are separate train and test programmes using rotation invariant features and rectangular features, giving `train_rotinv`, `test_rotinv`, `train_square`, and `test_square`.
- `features` : This contains code for extracting features from the images to use
in the random forests models.
- `forests` : This contains code for the random forest models used for analysing
the images, based on my [canopy](https://github.com/CPBridge/canopy) library.
- `filters` : This contains code for implementing the particle filters that link
estimates over time.
- `utilities` : This contains utility functions and definitions that are used
throughout the project.

Within the `scripts` directory, the scripts are placed into several subdirectories according to the task they perform:

- `masks` : Tool for creating image ROI masks.
- `datasets` : Scripts for creating and managing dataset files, which are used to train random forest models.
- `filter_fit` : Scripts for fitting particle filter models to training data.
- `run` : Scripts for coordination of cross-validation experiments with multiple parameter sets.
- `analysis` : Scripts for performing analysis of results and making plots and charts.
- `utilities` : These are python modules that are not intended to be called directly, but are used by scripts in the other directories.

## Models and Tasks

The main task of the video analysis software (implemented in the C++ part of the code) is to use a combination of particle filters and random forest models in order to track the following variables of interest within ultrasound videos of the fetal heart:

* Visibility of the heart in the image
* View plane classification (four-chamber view, left ventricular outflow tract view, three vessels view)
* Position of the heart in the image
* Orientation of the heart in the image
* Cardiac phase (point in the cardiac cycle)
* Locations of cardiac structures of interest

There are in fact two parallel pipelines, one which uses rotation invariant features for analysing the images, and another that uses traditional rectangular features. This difference also has implications for the design of the particle filters, hence the two different pipelines.

For each of the pipelines there are two executables built by the C++ code, one for training and one for testing. This gives `train_rotinv`, `train_square`, `test_rotinv`, and `test_square`.

Both of the testing executables can be run in a number of different modes, or 'problem types', which solve increasingly more complex tasks. There six modes (numbered 0 to 5), although the first two can only be used with `test_rotinv`. To control the mode, you pass it to the executable using the `-p` option (more on this later). The variables tracked in each of the different problem types is as follows:

Problem Type | Variables
---|---
0 | Class, Position, Visibility
1 | Class, Position, Visibility, Cardiac Phase
2 | Class, Position, Visibility, Orientation
3 | Class, Position, Visibility, Cardiac Phase, Orientation
4 | As in 3 plus cardiac structure locations (using a partitioned particle filter)
5 | As in 3 plus cardiac structure locations (using PCA for structure locations)

**Table 1** List of Problem Types

In order to perform these tasks, there are a number of different models that must be trained and provided to the testing executable. There are two categories of model: random forest models and filter models. These are required in different combinations depending on the features (rotinv or square) used and the problem type.

#### Random Forest Models

There are several different types of random forest model needed for different tasks. These are listed in the table below:

Name | Filename | Description
-|-|-|
classifier | `<base>.tr` | A random forest classifier trained to distinguish the different view classes and a background class.
jointOrientationRegressor | `<base>_ori.tr` | A random forest classifier in which each leaf node also contains a regressor for each class for predicting the orientation using equivariant features from the RIF feature extraction process.
circularRegressor | `<base>_phase<n>.tr` | A circular regression forest for predicting the cardiac phase of a heart of class `n`. There is one such model per view class.
jointOriPhaseRegressor | `<base>_phaseori<n>.tr` | A circular regression forest for predicting the cardiac phase of a heart of class `n` with an additional regressor in each leaf node for predicting the orientation using equivariant features from the RIF feature extraction process. There is one such model per view class.
classifier (structures) | `<base>_subs.tr` | A random forest classifier trained to distinguish the different structures from each other and from a background class.

**Table 2** List of random forest model types

Each random forest model is stored in a text file with the arbitrary extension `.tr`. This lists all the split parameters and node distribution parameters needed to fully define the random forest model (see the [canopy](https://github.com/CPBridge/canopy) library for more details).

The training executables train multiple forest models at once, according to the `-a` (orientation) and `-p` (cardiac phase) options. For the `train_rotinv` executable, the outputs produced are as follows:

Model | *none* | `-a` | `-p` | `-ap`
---|---|---|---|---
classifier | :white_check_mark: | | :white_check_mark: | :white_check_mark:
jointOrientationRegressor | | :white_check_mark: | |
circularRegressor | | | :white_check_mark: |
jointOriPhaseRegressor | | | | :white_check_mark:

**Table 3** Models produced by the `train_rotinv` executable with different options

For the `train_square` executable, the corresponding table is:

Model | *none* | `-p`
---|---|---
classifier | :white_check_mark: | :white_check_mark:
circularRegressor | | :white_check_mark:

**Table 4** Models produced by the `test_rotinv` executable with different options

Note that in both cases you are asked to provide a base file name, and all the output models share this same filename stem with a different ending according to the second column of **Table 2**. Also note that for the `train_square` case, a separate version of each model is trained for each of a number of different orientations. These are stored in separate files with the rotation index in the filename. Finally note that the classifiers used for structures are the same as used for the heart, except they are trained on different data. Therefore, use the same options but provide a different dataset file (more on this later).

Different problem types require different combinations of random forest models during testing. The following two tables show these requirements for the `test_rotinv` and `test_square` executables respectively.

Model | 0 | 1 | 2 | 3 | 4 | 5
---|---|---|---|---|---|---
classifier | :white_check_mark: | :white_check_mark: | | :white_check_mark: | :white_check_mark: | :white_check_mark:
jointOrientationRegressor | | | :white_check_mark: | | |
circularRegressor | | :white_check_mark: | | | |
jointOriPhaseRegressor | | | | :white_check_mark: | :white_check_mark: | :white_check_mark:
classifier ( structures) | | | | | :white_check_mark: | :white_check_mark:

**Table 5** Random forest models required for each problem type in the `test_rotinv` executable.

Model | 0 | 1 | 2 | 3 | 4 | 5
---|---|---|---|---|---|---
classifier | :x: | :x: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark:
circularRegressor | :x: | :x: | | :white_check_mark: | :white_check_mark: | :white_check_mark:
classifier (structures) | :x: | :x: | | | :white_check_mark: | :white_check_mark:

**Table 6** Random forest models required for each problem type in the `test_square` executable.

The :x: symbol means that the problem type is not implemented with rectangular features.


#### Particle Filter Models

The behaviour of the particle filters is governed by certain parameters that are stored in definition files. These definition files are shared by the rotinv and square executables (although the structure of the particle filters are slightly different in each case, they still require the same parameters).

The definition files consist of parameters that are fixed and specified by the user, as well as parameters that are fitted from training data using maximum likelihood estimation. The fixed parameters are stored in parameter files (default versions of these are found in the `scripts/filter_fit/params` directory), which are human readable to be easily editable. Each filter file is produced by a python script, which combines the fixed parameters in the parameter file with fitted parameters from a training set to produce the full filter definition file.

The following table lists the types of filter models, the scripts used to produce them, and the name of the default parameters file:

Name | Description | Created With | Parameters File
---|---|---|---
classOriFilter | Models how the view class, position and orientation variables behave. |  `fit_class_ori_filter.py` | `class_ori_filter_params`
phaseFilter | Models how the cardiac phase and cardiac phase rate variables behave. | `fit_phase_filter.py` | `phase_filter_params`
partitionedStructuresFilter | Models the structures' positions using a partitioned particle filter. | `fit_partitioned_structs_filter.py` | `structures_filter_params`
PCAStructuresFilter | Models the structures' positions using a principal component decomposition. | `fit_pca_structs_filter.py` | `structures_filter_params`

**Table 7** List of filter types, the scripts used to produce their definition files, and their default parameters files.

When running test using particle filters, the following filter files are required for the different problem types (apply for both `test_rotinv` and `test_square`).

| Filter Model | 0 | 1 | 2 | 3 | 4 | 5 |
|-----|----|----|----|---|----|----|
| classOriFilter | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| phaseFilter | | | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| partitionedStructuresFilter | | | | | :white_check_mark: | |
| PCAStructuresFilter | | | | | | :white_check_mark: |

**Table 8** Filter files required by different problem types

## Preparing Training Data

In order to use the tools in this repository, you will need a suitable set of fetal heart ultrasound videos. This section takes you through preparing this data for use with the software. Alternatively you can download and use pre-trained models for demonstration purposes (see the next section).

#### 1. Obtain Video Data

The framework expects a dataset of videos of the fetal heart which meet the following criteria:

- Be in the `.avi` format.
- Be placed in a single directory.
- Have names of the form `<subject-id>_<no>.avi` where `<subject-id>` is an arbitrary string that is the same for all videos captured from the same subject, and `<no>` is a positive integer number (of any length) that differentiates between the different videos for each subject. E.g. `subject001_3.avi` for the third video from subject `subject001` or `123456_10.avi` for the 10th video from subject `123456`. (There is no requirement that numbers are consecutive, just unique).
- Additionally you should place a text file called "patients" in the directory containing the videos and list the subjects (one per line) in that file.

#### 2. Create Region-of-Interest Masks

The videos should be accompanied by a `.png` image file that represents the ROI mask of the videos for a single subject. The ROI mask image should be non-zero within the ultrasound fan area (the area containing actual image information) and zero elsewhere (in blank areas and areas containing textual information etc that should be ignored by the image processing pipeline). The masks should be stored in a separate directory and have the format `<subject-id>_mask.png`. You can use the simple `maskcrop.py` tool in the `masks` directory to help you manually crop a polygonal mask from a video file.

```bash
$ ./maskcrop.py /path/to/videos/subject001_1.avi /path/to/masks/subject001_mask.png
```

#### 3. Annotate 'Global' Heart Variables

Each video should be accompanied by a *track file*, which contains the manual ground truth annotations of the heart's radius, visibility, location, orientation, view class, and cardiac phase in each frame of the video. These files are simple text files with the `.tk` extension and a certain layout. The tool for performing this annotation and producing the track files is in a separate repository at [CPBridge/heart_annotation_tool](https://github.com/CPBridge/heart_annotation_tool), whose documentation also gives further details about how to use it.

#### 4. Create Structures List

You only need to complete this part if you wish to track the locations of cardiac structures. You need to create a list of the structures you wish to track following the format described in the README of the [heart_annotation_tool](https://github.com/CPBridge/heart_annotation_tool) repository.

#### 5. Annotate Structure Locations

You only need to complete this part if you wish to track the locations of cardiac structures. Manual ground truth annotations are stored in a *structure track file* (`.stk`) file, which are produced with another tool in the [heart_annotation_tool](https://github.com/CPBridge/heart_annotation_tool) repository.

## Basic Workflow - Training and Testing a Model

This is an overview of a basic experimental workflow that takes you through training models and testing them on unseen videos. For more details, you will need to consult the relevant source files (most scripts and executables have help available if you pass the `-h` option). Where code examples are given, they pertain to GNU/Linux-like operating systems and may differ slightly on other platforms.

#### 1. Create Dataset Files

Dataset files are files defining a number of training examples to use when training a random forest model. Creating a dataset file involves finding all possible positive training patches from the track files, choosing a random subset of them, and choosing random negative/background examples. Scripts for performing these tasks are in the `datasets` directory.

To create a dataset file to train the random forests for 'global' variable estimates (heart view class, orientation, cardiac phase), you can use the `heartdatasets.py` script. You need to pass in the location of the directory containing the track (`.tk`) files (first positional argument), the number of view classes (`-v` option) present in the annotations (excluding the background class), the number of training examples you want for each of the positive classes (`-n` option), the names of any subjects you want to exclude from the training set (`-e` option), the name of the directory containing the mask file (`-m` option), and the name of the output file (`-o` option). For example,

```bash
$ ./heartdatasets.py /path/to/tracks/ -v 3 -n 5000 -m /path/to/masks/ -o /path/to/output/dataset/file
```

There are some other options here (jittering the training data for example), that you can look into (use `-h` for a list). One particularly important one is the `-f` option which allows you to specify that the dataset should not contain examples from the first few frames of a video (`-f 1` will omit the first frame and so on). If you wish to train models using motion features, you will need to use this as motion is not available in the first frame.

The file `substrdatasetsfromtracks.py` creates dataset files for structures. The usage is very similar to `heartdatasets.py`, but some additional information is needed. The directories containing both the `.stk` and `.tk` files must be passed as the first and second positional arguments respectively. The third positional argument is the fraction of the heart radius to use as the patch size for detecting structures. The fourth is the name of structures file that defines which structures are used. For example,

```bash
$ ./substrdatasetfromtracks.py /path/to/struct_tracks/ /path/to/tracks/ 0.5 /path/to/structure_list -n 5000 -m /path/to/masks/ -o /path/to/output/dataset/file
```

However, most of the time when you are using structures, you will want matched datasets to train matched models for the global variables and the structures. `heartandsubsdataset.py` wraps these two tasks into one for you, creating both a heart dataset and a structures dataset, where the latter has `_subs` appended to the name. Its usage is very similar to `substrdatasetsfromtracks.py` :

```bash
./heartandsubsdataset.py /path/to/struct_tracks/ /path/to/tracks/ 0.5 /path/to/structure_list -n 5000 -v 3 -m /path/to/masks/ -o /path/to/output/dataset/file
```

#### 2. Train Random Forest Models

Now you can train the random forests models using the dataset files produced in the previous stage. As described above, there are several different types of random model file needed for different problem types. Consult tables 5 and 6 above to work out which model files you need for the task you wish to perform and then 3 and 4 for the values of the `-a` and `-p` required to create them.

The process is slightly different depending on whether you are using the rotinv or square pipeline, but many of the options are the same.

For the rotinv pipeline (using rotation invariant features), we will use the `train_rotinv` executable to produce the models. This executable will be found in your build directory. The basic set of options (most of which are required) are:

* `-d <dataset>` - The name of the dataset file to use.
* `-v <video_directory>` - The directory containing all the videos that are mentioned in the dataset file.
* `-o <output_base_name>` - The base of the filename for all output models.
* `-f <feature_type>` - The image representation used for features. This must be one of `int` (image intensity), `grad` (image intensity gradient), `motion` (motion), or `mgof` (monogenic odd filter). Multiple may be listed, in which case the forest models may choose from any of the listed types during training.
* `-j`, `-k` and `-m` - These are the three parameters of the rotation invariant feature set (each should be integer valued). If you are using multiple feature types, you can list different values for each type (or list one, which will be used for all the feature types).
* `-a` and `-p` - These control which models are trained (see **Table 3**).
* `-n` - The number of trees in each forest model (integer valued)
* `-l` - The depth of trees (number of levels) in each forest model (integer valued)

There are further options that can be found with the help `-h` option.

The command will therefore look something like this:

```bash
$ ./train_rotinv -v /path/to/videos/directory/ -d /path/to/dataset_file -o /path/to/output/model -f grad motion -j 3 -k 3 -m 2 -a -p
```
For the `train_square` executable, most options are the same except that the `-j`, `-k`, `-m` options are missing as these relate to parameters specific to the rotation invariant features. Instead there are the following options:

* `-O` - Number of (equally spaced) orientations at which to train models.
* `-b` - Number of histogram bins for expansion of orientation histograms (only relevant for vector-valued features, i.e. not intensity).

Note that the training procedure may take a long time (hours) to run if you choose to use a large number of trees and/or levels, and/or a large number of orientations with `train_square`. It will also use all processors it has available.

To train forests for structure detection, the same executables should be used and a structure dataset should be passed with the `-d` option instead of a heart dataset. In order to use a structures detection forest model alongside heart detection and cardiac phase regression models, you should make sure that all training parameters relating to features match so that all the forest models are able to share the same feature extraction routines at test time (parameters such as the number of trees and tree depth can be different).

There is one exception to this, which is the `-x` option. This option allows features of different sizes to be used for structure detection and heart detection/analysis. Typically this is used to allow the image patch used to detect the structures to be smaller than that used to detect the heart. The goal of this option is the same in both the `train_square` and `train_rotinv` executables, but the way it works is different in each case. In  the case of rotation invariant features, the `-x` option allows you to choose a maximum value of the `-j` parameter (which indexes radial profiles) to use for the structures detection. This means that only the smaller, innermost radial profiles are included in the feature set for the structure models. In the case of the rectangular features, the `-x` parameter can be used to simply chooses a smaller patch size (in pixels) to use for the structures forest models.

In order to make sure that a structures forest can be matched to heart models and used in the same call to the testing executable, the name of the structures forest model must be the basename of the heart models with `_subs` appended.

#### 3. Create Filter Definition Files

The filter definition files are produced by the scripts in the `scripts/filter_fit/` directory, as in **Table 7**. Each requires some basic inputs (such as the directory containing the track files, where to place the output file, and the location of the parameters file to use) and has some more advanced options. You can also pass a list of excluded subject-ids to each (`-e` option) to make sure the test set is omitted. Check the options for each script by passing the `-h` flag to the relevant script.

Some basic examples are as follows:

```bash
# Fit a classOriFilter
./fit_class_ori_filter 3 /path/to/trackfiles/ params/class_ori_filter_params /path/to/output -e subject001

# Fit a phaseFilter
./fit_phase_filter /path/to/trackfiles/ /path/to/videofiles/ params/phase_filter_params /path/to/output -e subject001

# Fit a partitionedStructuresFilter
./fit_pca_structs_filter 3 /path/to/structure_trackfiles/ /path/to/heart_trackfiles/ /path/to/structure_list params/structures_filter_params /path/to/output -e subject001

# Fit a PCAStructuresFilter
./fit_partitioned_structs_filter 3 /path/to/structure_trackfiles/ /path/to/heart_trackfiles/ /path/to/structure_list params/structures_filter_params /path/to/output -e subject001

```

Note that the '3's relate to the number of viewing planes used in the annotations (excluding the background class).

#### 4. Test the Models

The testing executables (`test_rotinv` and `test_square`) take one video file, a number of random forest and filter models, and uses the models to estimate the value of the variables from each frame of the video. They should run at high frame rates, and can display the results on screen and/or write the frame-by-frame results to a file.

There are a few basic arguments that are required and a large number of other options. Most of these overlap for the the two executables.

The basic options for both executables are as follows:
* `-v` **Video**: The name of the video file to test.
* `-r` **Radius**: The radius of the heart in the video frames (in pixels). This is the only user-supplied data about the input video file.
* `-m` **Forest Model** The base name for the random forest model files. In order to find additional models (such as cardiac phase regression models etc), the extensions will be added according to the values in **Table 2**, so make sure all the necessary models exist for the problem type you are using and are placed in the same directory. All these models must also use exactly the same feature sets, which will always be the case if they were generated using a single run of the training executable, as is recommended.
* `-k` **Mask**: The name of the file to use as a mask of the ultrasound fan area (as described above).
* `-p` **Problem Type**: The problem type, as defined in **Table 1**. An integer between 0 and 5 (there is in fact a value of 6 in `test_rotinv` that also attempts to detect the fetal abdomen using a Hough forest, this feature sort of works but was mostly abandoned and hasn't been documented in this readme).

Other options of interest include:
* `-f` **Filtering**: Turn on particle filters. If this option is not set, the variables will be estimated independently in each frame. If set, a particle filter is used, and the filter definition files must also be provided (see `-z` option).
* `-z` **Filter Definition Files**: A list of all the filter definition files needed for filtering in this problem type. Exactly the right number of files must be provided in the correct order. To work this out, see **Table 8**, and list the names of the 1, 2, or 3 files necessary in the order they appear in that table. E.g. for problem type 5 you would list a classOriFilter definition file, then a phaseFilter definition file, and then a PCAStructuresFilter definition file.
* `-d` **Display Mode**: This controls what is displayed on screen during testing. There are five options, numbered 0 to 4:
	* **0 - 'None'**: Nothing is displayed to the screen. This can increase processing speed quite significantly.
	* **1 - 'Detection'**: A simple screen showing the input frame and the variable estimates is displayed. In this display, the position of the circle shows the heart position, the radial line shows the orientation, the colour of the circle shows the view plane classification (*cyan* four chamber view, *green* left ventricular outflow tract view, *yellow* three vessel view) and the moving arrowhead shows cardiac phase (moving out -> systole, moving in -> diastole).
	* **2 - All**: Several windows appear showing different information. One shows the detection, as in mode 1. Then there is one additional windows per view class. If particle filtering is being used, these show information about each particle that currently has the relevant view imposed over the input image (where relevant, particle colour indicates cardiac phase, arrowhead direction indicates particle orientation). If particle filtering is not used, the windows show the detection score for each view class from the detection forest model. If structures are being tracked, their is one window per structure instead of per view class.
	* **3 - 'Ground Truth'**: Shows the detection window as in mode 1, and also shows the ground truth in a separate window, displayed in the same way. The track file(s) must be provided (`-g` and `-i` options) to enable this.
	* **4 - 'No Background'**: The same as mode 2, except that the particle or detection forest output are shown over a black image instead of being superimposed over the input image.
* `-o` **Output**: Write the frame by frame results to this output file. This is a basic text file, with each line corresponding to one frame in the video. There are scripts elsewhere in the repository designed to work with them.
* `-c` **Record**: Record the displayed windows to a `.avi` file with the provided name.
* `-u` **Pause**: Pause between each frame (waits for any key to be pressed before continuing).
* `-n` **Trees (Detection)**: Number of trees to use in the detection forest (must be less than or equal to the number of trees in the trained model file).
* `-l` **Tree Levels (Detection)**: Number of levels to use in the trees to in the detection forest (must be less than or equal to the number of levels in the trained model file).
* `-N` **Trees (Phase Regression)**: Number of trees to use in the cardiac phase regression forest (must be less than or equal to the number of trees in the trained model file).
* `-L` **Tree Levels (Phase Regression)**: Number of levels to use in the trees to in the cardiac phase regression forest (must be less than or equal to the number of levels in the trained model file).
* `-S` **Trees (Structures)**: Number of trees to use in the structures detection forest (must be less than or equal to the number of trees in the trained model file).
* `-T` **Tree Levels (Structures)**: Number of levels to use in the trees to in the structures detection forest (must be less than or equal to the number of levels in the trained model file).
* `-Q` **Particles**: Number of particles to use in the particle filter.
* `-g` **Ground Truth Track File**: A track file (`.tk`) for the video that is used to display the ground truth in display mode 3.
* `-i` **Ground Truth Track File (Structures)**: A structures track file (`.stk`) for the video that is used to display the ground truth in display mode 3.
* `-P` **Use Ground Truth Position**: This refers only to testing without filtering. If set, the estimates for the orientation and cardiac phase are calculated at the ground truth position of the heart, rather than the detection position of the heart (which is the default). To use this, the track file to use for the ground truth must be provided with the `-g` option.

Extra options specific to `test_rotinv` include:
* `-R` **RIF Calculation Method**: Method used for calculating rotation invariant features. Allowed options are "spatial" or "s" for spatial domain convolutions, "frequency" or "f" for frequency domain calculations or "auto" or "a" for automatic choice of method.
* `-C` **RIF Coupling Method**: Method used for coupling calculations for rotation invariant features. Allowed options are "element-wise" or "e" for element-wise coupling, "vectorised" or "v" for vectorised coupling (entire images at a time) or "auto" or "a" for automatic choice of method.

#### 5. Parse results

The `parse_output.py` script in the `scripts/analysis` directory takes the results file recorded from the testing executables (using the `-o` option) and compare them to the track file to work out the accuracy of the automatic estimates. To get a summary printed to the terminal you can use something like this:

```bash
$ ./parse_output.py /path/to/results/file /path/to/track/directory 0 0 0.25 -s
```

The first two arguments here are the results file to be analysed and the directory where the relevant track file is stored (the correct track file will be automatically selected to match the name of the video that the test was performed on). The next two arguments relate to functionality that we are to using here, and are therefore set to zero (this script can also be used to generate new datasets including automatically selected 'hard' negatives). The fifth argument is the distance threshold within which a detection should be considered 'correct', written as a fraction of the annotated radius of the heart (here we are using a quarter). The `-s` flag causes a human readable summary to be printed to screen.

## Using Pre-trained Models

Some pre-trained models, trained using the videos in the dataset used for my thesis, can be found in a [separate repository](https://github.com/CPBridge/fetal_heart_models). You can use these models to run the algorithm on your own fetal heart ultrasound videos, but bear in mind that the models may not generalise well to different datasets (due to differences in probe parameters, scanning protocols etc).

Suppose you had an example video called `video.avi`, in which the fetal heart appeared with radius 100 pixels, a corresponding mask image `video_mask.png` is available, and the fetal heart models repository has been cloned into your home directory `~`. The following examples will demonstrate how to perform some basic tests on your video with different models:

- Use a model based on rotation invariant features with particle filtering to estimate global heart variables (this is is a good set of parameters to use to get fast, reasonably accurate results):
```bash
$ ./test_rotinv -v /path/to/video.avi -r 100 -k /path/to/video_mask.png -p 3 -m ~/fetal_heart_models/forest_models/rotinv/rotinv_j4k3m2 -Rf -Ce -n32 -l12 -N16 -L8 -f -z ~/fetal_heart_models/filter_models/class_ori_filter ~/fetal_heart_models/filter_models/phase_filter
```
- Use a model with rectangular features to perform the same task:
```bash
$ ./test_square -v /path/to/video.avi -r 100 -k /path/to/video_mask.png -p 3 -m ~/fetal_heart_models/forest_models/rec/rec -n32 -l12 -N16 -L8 -f -z ~/fetal_heart_models/filter_models/class_ori_filter ~/fetal_heart_models/filter_models/phase_filter
```

- Use a model based on rotation invariant features with particle filtering to estimate global variables and additionally structure locations using a PCA-based model:
```bash
$ ./test_rotinv -v /path/to/video.avi -r 100 -k /path/to/video_mask.png -p 5 -m ~/fetal_heart_models/forest_models/rotinv/rotinv_j4k3m2 -Rf -Ce -n32 -l12 -N16 -L8 -S16 -T10 -f -z ~/fetal_heart_models/filter_models/class_ori_filter ~/fetal_heart_models/filter_models/phase_filter ~/fetal_heart_models/filter_models/pca_structs_filter
```

- Use a model with rectangular features to estimate global variables and additionally structure locations using a partitioned model:
```bash
./test_square -v /path/to/video.avi -r 100 -k /path/to/video_mask.png -p 4 -m ~/fetal_heart_models/forest_models/rec/rec -n32 -l12 -N16 -L8 -S16 -T10 -f -z ~/fetal_heart_models/filter_models/class_ori_filter ~/fetal_heart_models/filter_models/phase_filter ~/fetal_heart_models/filter/models/partitioned_structs_filter
```

For a more detailed summary of the options for the test executables, see the previous section.

## Cross-Validation Experiment Workflow

There are also a number of scripts in the repository to facilitate large scale experiments to compare several different parameter sets. Because of the limited of availability of data, these have been implemented to perform a leave-one-subject-out cross-validation. This means that for each subject in the dataset, a model is trained using all the training data that does *not* originate from that subject. This model is then tested on that subject. This is performed for all subjects in the dataset, and results are aggregated across all subjects.

The parameters of interest are divided into two sets: *training parameters*, which govern the training of the forest models, and *testing parameters*, which control various aspects of the testing process. You can define multiple sets of training and testing parameters in special files, and use scripts to perform tests on all of them.

Note that performing cross-validation experiments with several parameters can take a very long time! The training and testing stages in this workflow can each take several days to complete on a modern desktop PC and will use most of the available processor during that time (but usually not too much RAM unless you are training huge forests).

#### 1. Create Dataset Files

To train the models in a cross-validation experiment, we first need datasets files for each cross-validation fold, i.e. excluding each of the subjects. This is done exactly as described in the Basic Workflow section above, with the addition of passing the `-c` flag option to automatically generate one dataset file for each fold (each fold excludes a single subject, according to the subject-id part of the video's file name, and the relevant dataset file's name indicates which subject was excluded by appending `_ex<subject-id>`). This works in the same way for each of the relevant scripts.

#### 2. Create Filter Definition Files

This is also done in the same way as in the basic flow, with the addition of passing the `-c` to each of the scripts that generate the filters. This produces a number of filter files that each exclude one subject identified by appending `_ex<subject-id>` to the file name to indicate which subject was excluded.

#### 3. Create Training Experiment Files

A *training experiment file* lists one or more sets of parameters for training forest models using the training executables. You should create these manually with a text editor.

The structure is very straightforward. Each line of the file relates to one set of training parameters. The first word on the line (before the first white space) is an arbitrary name to identify the parameter set. The rest of the line is parameters for the relevant training executable written exactly as they would be passed to the executable.

There are however some options you should **not** specify, because they will be controlled automatically when you run the experiment file (next step). These are the: the name of the video (`-v`), the name of the output file (`-o`) and the name of the dataset file (`-d`).

A basic example of a training experiment file for `train_rotinv` with a few different feature types and J, K, M parameters is shown below:

```
int33 -f int -j 3 -k 3 -pa -n 32 -l 12
grad332 -f grad -j 3 -k 3 -m2 -pa -n 32 -l 12
grad543motion332 -f grad motion -j 5 3 -k 4 3 -m 3 2 -pa -n 32 -l 12
```

#### 4. Run Training Experiment File

In order to train all the models listed in the training experiment file, you use the `train_experiment_file.py` script in the `scripts/run/` directory. The required arguments for this script are (in order):

- The full path to the training binary on your system (i.e. the full path of either the `train_rotinv` or `train_square` binary).
- The path to the training experiment file you want to use (from the previous step).
- The directory containing the video files
- The directory containing the datasets for the cross-validation
- The base name of the datasets within the above directory (i.e. without the `_ex<subject-id>` part).
- The directory where the trained models should be placed.

The command will look something like this:

```bash
$ ./train_experiment_file /path/to/build/directory/train_rotinv /path/to/training/experiment/file /path/to/video/directory/ /path/to/dataset/directory/ dataset_name /path/to/model/directory/
```

This will run the training routine for every cross-validation fold and every set of training parameters in the training experiment file. The output models will be placed in the specified directory with models trained with each set of parameters placed within their own subdirectories with the same name as the parameter set. The models for the different folds are again identified by the addition of `_ex<subject-id>` to the model name.

Note that this will take a long time (days) to run in many cases, depending of course of how many subjects you have, how many sets of training parameters you use, and the parameters you use (particularly the number of trees to train). If for whatever reason this gets interrupted, you can resume where you left off using the `-i` option, which will not repeat the training for any model file that already exists in the specified output directory.

#### 5. Create Testing Experiment Files

Multiple sets of testing parameters (parameters for the testing executables) are specified in *testing experiment files* just like the training parameters. These files work in an almost identical way: each line contains one set of parameters, the first part of the line names that parameter set, and the rest are parameters to the testing executables exactly as they would be passed on the command line.

Again, some of the parameters are automatically controlled for you by the script (next step) and therefore should **not** be specified again in the file. This includes the following:

* `-v` : the video file
* `-m` : the random forest model file stem
* `-o` : results output file
* `-r` : detection radius
* `-d` : display mode
* `-g` : ground truth track file
* `-k` : mask file
* `-z` : filter definition files

A simple example looks like this:

```
n8l8freq -n 8 -l 8 -N 8 -L 8 -Rf -p 3
n16l8freq -n 16 -l 8 -N 16 -L 8 -Rf -p 3
n32l8freq -n 32 -l 8 -N 32 -L 8 -Rf -p 3
```

#### 6. Run Testing Experiment File

To run a testing experiment file, use the `test_experiment_file.py` script in the `scripts/run/` directory. You you provide both a training experiment file and a testing experiment file, and each set of testing parameters is run on the models trained using each set of training parameters in the training experiment file. This means that if the training and testing experiment files are both quite long, you will be running a very large number of cross-validation experiments -- use with caution!

The required arguments for this script are (in order):

- The full path to the testing binary on your system (i.e. the full path of either the `test_rotinv` or `test_square` binary).
- The path to the testing experiment file you want to use (from the previous step).
- The path to the training experiment file you want to use. It is assumed that the models exist for all the training parameter sets in the training experiment file.
- The directory containing the video files
- The top-level directory containing the trained models (i.e. the same one specified to the `train_experiment_file`).
- The directory where the results should be placed.
- The directory containing the track files.
- The directory containing the mask image.

There are a number of optional parameters, including:
- `-n` **Number of trials**: Since the particle filters introduce stochasticity into the output, each run with the same set of parameters will give different results. You can therefore use this option to repeat each test multiple times, and store each set of results (the trial number is appended to the end of the results file name). Note that is no point repeating trials if you are not using particle filtering, because the output should be exactly the same each time.
- `-i` **Ignore exisiting**: As in `train_experiment_file.py`, when this is set trials for which results files exists will not be performed again. It therefore allows you to pick up where a previous set of experiments left off.
- `-z` **Filter Definition Extensions**: This is the base name of the filter definition files to use (i.e. without the `_ex<subject-id>` extension). Note that if you use filter, the results will be placed in a further subdirectory having the base name of the filter.
- `-v` **Verbose**: Displays each command before execution.

It's best to estimate roughly how long this will take first, based on the fact that testing each video is tested in approximately real time and each is tested `num_trials` times for every combination of training and testing experiments.

After this has completed, the results directory will be populated with the results in a hierarchy of sub-directories. At the top of this hierarchy, the results are grouped by their testing parameters, and then at the lower level by their training parameters.

#### 7. Summarise Experiments

Before performing analysis that compares different parameter set, it is necessary to first summarise the accuracy of each individual experiment (one set of training and testing parameters across all the tests performed in the cross-validation fold). The script `summarise.py` in the `scripts/analysis` directory is used to perform this task. It takes the same training and testing experiment files used by the `train_experiment_file.py` and `test_experiment_file.py` scripts, and summarises each experiment by internally calling the `parse_output.py` script on each results file and storing the results in a file called `summary` alongside the results files.

The positional arguments for this script are:
- The testing experiment file
- The training experiment file
- The top-level results directory (same as the one passed to `test_experiment_file.py`).

Other options include:
- `-r` **Radius threshold**: The threshold distance between the detection and the ground truth that is considered a corrent detection, written as a proportion of the heart radius (default 0.25).
- `-m` **Matching Pattern**: If you only wish to use a subset of the result files, you can specify a match pattern (in the format of a python glob) to decide which to include.
- `-t` **Track Directory**: The directory containing the track files (you must specify this unless you edit the source file to change the default).
- `-s` **Structs Track Directory**: Directory containing the structure track files if necessary.


#### 8. Make Plots

Once you have the summary files, you can use a number of scripts in the `scripts/analysis/` directory to make various plots to demonstrate the effects of different parameters. These scripts all share a similar format, and require you to pass in the names of the training and testing experiment files to use, and the top-level results directory containing the results files and the summary files produced in the last section. The scripts then parse the training and testing files in order to work out what parameter values were used, then read the summary files to get values for various performance metrics, and makes some plots based on the results.

The following scripts are available, you can use their help functions (`-h`) to find what options are available in each case:

- `accuracy_vs_time.py`: Draws three scatter plots with classification/detection error, orientation error and cardiac phase error respecitvely on the *x*-axes, and calculation time per frame on the *y*-axes for each training experiment. This demonstrates the trade off between accuracy and speed when choosing different feature sets.  
- `confusion.py`: Draws coloured confusion matrices for each training experiment. These show which view classes are mistaken for each other.
- `feature_set_bars.py`: Draws a bar chart of accuracy values for different training feature sets of rotation invariant features, with grouped bars for the different feature sets (no coupling, coupling and extra coupling) based on the same base features.
- `forest_test_graphs.py`: Draws three scatter plots with classification/detection error, orientation error and cardiac phase error respecitvely on the *x*-axes, and calculation time per frame on the *y*-axes for a number of different test-time forest configurations (different forest sizes and forest depths).
- `particles_graphs.py`: Draws three scatter plots with classification/detection error, orientation error and cardiac phase error respecitvely on the *x*-axes, and calculation time per frame on the *y*-axes for a number of different numbers of particles at test time.
- `time_bars.py`: Plot the time taken per frame for different rotation invariant feature calculation methods at test time.
- `structure_bar_plot.py`: Plot the localisation accuracy for different cardiac structures.
- `sensitivity.py`: Plot showing the trade-off between true positive detection rate and false positive rate for experiments without particle filtering, as the detection threshold is changed.
- `sensitivity_filtered.py`: Plot showing the trade-off between true positive detection rate and false positive rate for experiments with particle filtering, as the parameters governing the behaviour of the 'hidden' state of the particles are changed.

## References

The code in this repository implements the experiments conducted in my DPhil (PhD) thesis:
- C.P. Bridge, “Computer-Aided Analysis of Fetal Cardiac Ultrasound Videos”, DPhil Thesis, University of Oxford, 2017. Available on [my website](https://chrisbridge.science/publications.html).

The following publications also make use of this code:

- C.P. Bridge, C. Ioannou, and J.A. Noble, “Automated Annotation and Quantitative Description of Ultrasound Videos of the Fetal Heart”, *Medical Image Analysis 36* (February 2017) pp. 147-161. Open access available [here](http://dx.doi.org/10.1016/j.media.2016.11.006).
- C.P. Bridge, Christos Ioannou, and J.A. Noble, “Localizing Cardiac Structures in Fetal Heart Ultrasound Video”, *Machine Learning in Medical Imaging Workshop, MICCAI, 2017*, pp. 246-255. Original article available [here](https://link.springer.com/chapter/10.1007/978-3-319-67389-9_29). Authors' manuscript available on [my website](https://chrisbridge.science/publications.html).
