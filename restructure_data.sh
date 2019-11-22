#!/usr/bin/env bash


ff_download_script=$1
IFS=$'\n'

echo "Downloading FaceForensics++ data.. (this step takes a while)"
mkdir datasets
mkdir datasets/T_deepfake
bash download_T.sh $ff_download_script

echo "Restructuring FaceForensics++ data.."
cd datasets
mkdir base_deepfake
mkdir base_deepfake/real
mkdir base_deepfake/fake
mkdir base_deepfake/val
mkdir base_deepfake/val/real
mkdir base_deepfake/val/fake

cp T_deepfake/original_sequences/youtube/raw/videos/* base_deepfake/real/
cp T_deepfake/original_sequences/actors/raw/videos/* base_deepfake/real/

for file in T_deepfake/manipulated_sequences/Deepfakes/raw/videos/*.mp4; do cp $file base_deepfake/fake/deepfake_$(basename $file); done
for file in T_deepfake/manipulated_sequences/Face2Face/raw/videos/*.mp4; do cp $file base_deepfake/fake/face2face_$(basename $file); done
for file in T_deepfake/manipulated_sequences/FaceSwap/raw/videos/*.mp4; do cp $file base_deepfake/fake/faceswap_$(basename $file); done
for file in T_deepfake/manipulated_sequences/NeuralTextures/raw/videos/*.mp4; do cp $file base_deepfake/fake/neuraltextures_$(basename $file); done
for file in T_deepfake/manipulated_sequences/DeepFakeDetection/raw/videos/*.mp4; do cp $file base_deepfake/fake/deepfakedetection_$(basename $file); done

mv base_deepfake/real/07__talking_against_wall.mp4 base_deepfake/val/real/
mv base_deepfake/real/01__kitchen_pan.mp4 base_deepfake/val/real/

mv base_deepfake/fake/deepfakedetection_06_04__walking_outside_cafe_disgusted__ZK95PQDE.mp4 base_deepfake/val/fake/
mv base_deepfake/fake/deepfakedetection_09_03__kitchen_pan__8DTEGQ54.mp4 base_deepfake/val/fake/

mv base_deepfake/real/339.mp4 base_deepfake/val/real/
mv base_deepfake/real/878.mp4 base_deepfake/val/real/

mv base_deepfake/fake/*_339_*.mp4 base_deepfake/val/fake/
mv base_deepfake/fake/*_878_*.mp4 base_deepfake/val/fake/

mv base_deepfake/real/8* base_deepfake/val/real/
mv base_deepfake/real/71* base_deepfake/val/real/
mv base_deepfake/real/992* base_deepfake/val/real/

echo "Downloading Youtube pre-structured data.."
wget https://deepfake-detection.s3.amazonaws.com/augment_deepfake.tar.gz
tar -xvzf augment_deepfake.tar.gz

echo "Extracting FaceForensics++ data frames.."
cd base_deepfake/real
mkdir frames
for f in *.mp4; do ffmpeg -r 3 -i $f -r 1 "frames/${f%.*}_%03d.png"; done
cd ../fake
mkdir frames
for f in *.mp4; do ffmpeg -r 3 -i $f -r 1 "frames/${f%.*}_%03d.png"; done
cd ../val/real
mkdir frames
for f in *.mp4; do ffmpeg -r 3 -i $f -r 1 "frames/${f%.*}_%03d.png"; done
cd ../fake
mkdir frames
for f in *.mp4; do ffmpeg -r 3 -i $f -r 1 "frames/${f%.*}_%03d.png"; done

echo "Extracting Youtube data frames.."
cd ../../../augment_deepfake/real
mkdir frames
for f in *.mp4; do ffmpeg -r 1 -i $f -r 1 "frames/${f%.*}_%03d.png"; done
cd ../fake
mkdir frames
for f in *.mp4; do ffmpeg -r 1 -i $f -r 1 "frames/${f%.*}_%03d.png"; done
cd ../val/real
mkdir frames
for f in *.mp4; do ffmpeg -r 1 -i $f -r 1 "frames/${f%.*}_%03d.png"; done
cd ../fake
mkdir frames
for f in *.mp4; do ffmpeg -r 1 -i $f -r 1 "frames/${f%.*}_%03d.png"; done

cd ../../../

echo "Creating combined data (FaceForensics++ + Youtube).."
mkdir both_deepfake
cp -r augment_deepfake/* both_deepfake/
cp -r base_deepfake/* both_deepfake/

real_paper_train=$(ls base_deepfake/real/frames | wc -l)
fake_paper_train=$(ls base_deepfake/fake/frames | wc -l)
real_paper_val=$(ls base_deepfake/val/real/frames | wc -l)
fake_paper_val=$(ls base_deepfake/val/fake/frames | wc -l)

real_yt_train=$(ls augment_deepfake/real/frames | wc -l)
fake_yt_train=$(ls augment_deepfake/fake/frames | wc -l)
real_yt_val=$(ls augment_deepfake/val/real/frames | wc -l)
fake_yt_val=$(ls augment_deepfake/val/fake/frames | wc -l)

echo "prepared data stats"
echo "                  |  paper_train    |    paper_test     |      youtube_train      |      youtube_test      "
echo "------------------+-----------------+-------------------+-------------------------+------------------------"
echo "      real        |      $real_paper_train      |       $real_paper_val        |           $real_yt_train          |          $real_yt_val          "
echo "------------------+-----------------+-------------------+-------------------------+------------------------"
echo "      fake        |       $fake_paper_train      |        $fake_paper_val       |           $fake_yt_train         |          $fake_yt_val          "
echo "------------------+-----------------+-------------------+-------------------------+------------------------"

echo "All done. Happy training!"