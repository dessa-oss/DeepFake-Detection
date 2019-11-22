![parallel coordinates plot](/images/parcoords.gif)

# Visual DeepFake Detection

In our recent [article](add link), we make the following contributions:
* We show that the model proposed in current state of the art in video manipulation (FaceForensics++) does not generalize to real-life videos randomly 
collected 
from Youtube.
* We show the need for the detector to be constantly updated with real-world data, and propose an initial solution in hopes of solving deepfake video detection.

Our Pytorch implementation, conducts extensive experiments to demonstrate that the datasets produced by Google and detailed in the FaceForensics++ 
paper are not sufficient for making neural networks generalize to detect real-life face manipulation techniques. It also provides a current solution for such
 behavior which relies on adding more data. 
 
Our Pytorch model is based on a pre-trained ResNet18 on Imagenet, that we finetune to solve the deepfake detection problem.
We also conduct large scale experiments using Dessa's homemade scheduler + experiment manger [Atlas](www.atlas.dessa.com/)

To check the results we achieved in the article, please check our [interactive results UI](http://deepfake-detection.dessa.com/projects)

## Setup 
0. install [nvidia-docker](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0))
00. install [ffmpeg](https://www.ffmpeg.org/download.html) or `sudo apt install ffmpeg`
1. Git Clone this repository.
2. If you haven't already, install [Foundations Atlas Community Edition](https://www.atlas.dessa.com/?u=dessafake).
3. Once you've installed Foundations Atlas, activate your environment if you haven't already, and navigate to your project folder

That's it, You're ready to go!

Note: To run the code, your system should meet the following requirements: 
RAM >= 32GB , GPUs >=1

## Datasets
Half of the dataset used in this project is from the [FaceForensics](https://github.com/ondyari/FaceForensics/tree/master/dataset) deepfake detection dataset
. To download this data, please make sure to fill out the [google form](https://github.com/ondyari/FaceForensics/#access) to request access to the data.

For the dataset that we collected from Youtube, it is accessible on [S3](ttps://deepfake-detection.s3.amazonaws.com/augment_deepfake.tar.gz) for download.

To automatically download and restructure both datasets, please execute:

```
bash restructure_data.sh faceforensics_download.py
```

Note: You need to have received the download script from FaceForensics++ people before executing the restructure script.

Note2: We created the `restructure_data.sh` to do a split that replicates our exact experiments avaiable in the UI above, please feel free to change the 
splits as you wish.

## Walkthrough

Before starting to train/evaluate models, we should first create the docker image that we will be running our experiments with. To do so, we already prepared
 a dockerfile to do that inside `custom_docker_image`. To create the docker image, execute the following commands in terminal:
 
 ```
 cd custom_docker_image
 nvidia-docker build . -t atlas_ff
 ```
 
Note: if you change the image name, please make sure you also modify line 16 of `job.config.yaml` to match the docker image name.

Inside `job.config.yaml`, please modify the data path on host from `/media/biggie2/FaceForensics/datasets/` to the absolute path of your `datasets` folder.

The folder containing your datasets should have the following structure:

```
datasets
├── augment_deepfake        (2)
│   ├── fake
│   │   └── frames
│   ├── real
│   │   └── frames
│   └── val
│       ├── fake
│       └── real
├── base_deepfake           (1)
│   ├── fake
│   │   └── frames
│   ├── real
│   │   └── frames
│   └── val
│       ├── fake
│       └── real
├── both_deepfake           (3)
│   ├── fake
│   │   └── frames
│   ├── real
│   │   └── frames
│   └── val
│       ├── fake
│       └── real
├── precomputed             (4)
└── T_deepfake              (0)
    ├── manipulated_sequences
    │   ├── DeepFakeDetection
    │   ├── Deepfakes
    │   ├── Face2Face
    │   ├── FaceSwap
    │   └── NeuralTextures
    └── original_sequences
        ├── actors
        └── youtube
```

Notes:
* (0) is the dataset downloaded using the FaceForensics repo scripts
* (1) is a reshaped version of FaceForensics data to match the expected structure by the codebase. subfolders called `frames` contain frames collected using 
`ffmpeg`
* (2) is the augmented dataset, collected from youtube, available on s3.
* (3) is the combination of both base and augmented datasets.
* (4) precomputed will be automatically created during training. It holds cashed cropped frames.

Then, to run all the experiments we will show in the article to come, you can launch the script `hparams_search.py` using:

```bash
python hparams_search.py
```

## Using the Pre-trained Model 

To re-use some of the models we pre-trained in our experiments, it is possible to go on the 
[interactive results UI](http://deepfake-detection.dessa.com/projects), and pick which experiment you want to download the model for:

![open artifacts](/images/open_artifacts.png)

Download the model file, stored as `best_model.pth`.

![download model](/images/download_model.png)

Note: Some of the experiments don't have pre-trained models as we didn't introduce saving checkpoints at that point in time.

## Help improve this technology

Please feel free to fork this work and keep pushing on it.

If you also want to help improving the deepfake detection datasets, please share your real/forged samples at foundations@dessa.com