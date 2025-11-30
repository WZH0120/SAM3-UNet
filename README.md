
## [SAM3-UNet: Simplified Adaptation of Segment Anything Model 3](https://arxiv.org/)

## Introduction
![framework](./sam3unet.jpg)In this paper, we introduce SAM3-UNet, a simplified variant of Segment Anything Model 3 (SAM3), designed to adapt SAM3 for downstream tasks at a low cost. Our SAM3-UNet consists of three components: a SAM3 image encoder, a simple adapter for parameter-efficient fine-tuning, and a lightweight U-Net-style decoder. Preliminary experiments on multiple tasks, such as mirror detection and salient object detection, demonstrate that the proposed SAM3-UNet outperforms the prior SAM2-UNet and other state-of-the-art methods, while requiring only 3.6 GB of GPU memory during training with a batch size of 1.

[微信交流群](https://github.com/WZH0120/SAM2-UNet/blob/main/wechat.jpg)

## Clone Repository
```shell
git clone https://github.com/WZH0120/SAM3-UNet.git
cd SAM3-UNet/
```

## Prepare Datasets
You can refer to the following repositories and their papers for the detailed configurations of the corresponding datasets.
- Salient Object Detection. Please refer to [SALOD](https://github.com/moothes/SALOD).
- Mirror Detection. Please refer to [HetNet](https://github.com/Catherine-R-He/HetNet).

## Requirements
Please refer to [SAM 3](https://github.com/facebookresearch/sam3).

## Training
If you want to train your own model, please download the pre-trained sam3.pt according to [official guidelines](https://github.com/facebookresearch/sam3). After the above preparations, you can run `train.sh` to start your training.

## Testing
Our pre-trained models and prediction maps can be found at [Google Drive](https://drive.google.com/drive/folders/1J__gz-ZlTnpmDp3yGiohA58K4Ppm2ruQ). Also, you can run `test.sh` to obtain your own predictions.

## Evaluation
After obtaining the prediction maps, you can run `eval.sh` to get the quantitative results. For the evaluation of mirror detection, please refer to `eval.py` in [HetNet](https://github.com/Catherine-R-He/HetNet) to obtain the results.

## Citation and Star
Please cite the following paper and star this project if you use this repository in your research. Thank you!
```
```

## Acknowledgement
[SAM 3](https://github.com/facebookresearch/sam3)&emsp;[SAM2-UNet](https://github.com/WZH0120/SAM2-UNet/)
