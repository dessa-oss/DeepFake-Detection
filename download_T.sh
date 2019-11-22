#!/usr/bin/env bash


faceforensics_script=$1

python $faceforensics_script datasets/T_deepfake -d original -c raw -t videos -n 25 --server CA
python $faceforensics_script datasets/T_deepfake -d Deepfakes -c raw -t videos -n 5 --server CA
python $faceforensics_script datasets/T_deepfake -d Face2Face -c raw -t videos -n 5 --server CA
python $faceforensics_script datasets/T_deepfake -d FaceSwap -c raw -t videos -n 5 --server CA
python $faceforensics_script datasets/T_deepfake -d NeuralTextures -c raw -t videos -n 5 --server CA
python $faceforensics_script datasets/T_deepfake -d DeepFakeDetection -c raw -t videos -n 5 --server CA
