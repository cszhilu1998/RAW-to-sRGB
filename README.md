# RAW-to-sRGB (ICCV 2021)

PyTorch implementation of [**Learning RAW-to-sRGB Mappings with Inaccurately Aligned Supervision**](https://openaccess.thecvf.com/content/ICCV2021/html/Zhang_Learning_RAW-to-sRGB_Mappings_With_Inaccurately_Aligned_Supervision_ICCV_2021_paper.html) 

## 1. Framework

<p align="center"><img src="./figs/framework.png" width="95%"></p>
<p align="center">Figure 1: Illustration of the proposed joint learning framework.</p>

## 2. Results

<p align="center"><img src="./figs/results.png" width="95%"></p>
<p align="center">Figure 2: Example of data pairs of ZRR and SR-RAW datasets, where clear spatial misalignment can be observed with the reference line. With such inaccurately aligned training data, PyNet [22] and Zhang et al. [62] are prone to generating blurry results with spatial misalignment, while our results are well aligned with the input.</p>

## 3. Preparation

- **Prerequisites**
    - Python 3.x and PyTorch 1.6.
    - OpenCV, NumPy, Pillow, CuPy, colour_demosaicing, tqdm, lpips, scikit-image and tensorboardX.

- **Dataset**
    - [Zurich RAW to RGB dataset](https://docs.google.com/forms/d/e/1FAIpQLSdH6Pqdlu0pk2vGZlazqoRYwWsxN3nsLFwYY6Zc5-RUjw3SdQ/viewform). It can also be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1NpJacIOowSsfsIgJZrmjKg?pwd=8sq1) or [TeraBox](https://1024terabox.com/s/1_rYqk83J2RTFcva5KqgTVg).
    - [Preprocessed SR-RAW Dataset](https://drive.google.com/drive/folders/1hpLG1ksFV_76ZNrUg9XGvSotMvX9tV_Z?usp=sharing). Note that here we preprocessed the original SR-RAW dataset according to the [code](https://github.com/ceciliavision/zoom-learn-zoom/blob/master/demo_rawrgb_pair.ipynb). You can also download the original SR-RAW dataset [here](https://drive.google.com/drive/folders/1UHKEUp77tiCZ9y05JtP6S9Tfo2RftK8m).
       
## 4. Quick Start


### 4.1 Pre-trained models

- [The pre-trained models](https://drive.google.com/drive/folders/12eMNZh-A9E4bVup-Kcy_F0_c0IToUyw7?usp=sharing) can be downloaded. You need to put them in the `RAW-to-sRGB/ckpt/` folder.

### 4.2 Training

- Zurich RAW to RGB dataset 

    [`sh train_zrr.sh`](train_zrr.sh)

- SR-RAW Dataset
    
    [`sh train_srraw.sh`](train_srraw.sh)

### 4.3 Testing

- Zurich RAW to RGB dataset 

    [`sh test_zrr.sh`](test_zrr.sh)

- SR-RAW Dataset
    
    [`sh test_srraw.sh`](test_srraw.sh)

### 4.4 Note

- You can specify which GPU to use by `--gpu_ids`, e.g., `--gpu_ids 0,1`, `--gpu_ids 3`, `--gpu_ids -1` (for CPU mode). In the default setting, all GPUs are used.
- You can refer to [options](./options/base_options.py) for more arguments.

## 5. Citation
If you find it useful in your research, please consider citing:

    @inproceedings{RAW-to-sRGB,
        title={Learning RAW-to-sRGB Mappings with Inaccurately Aligned Supervision},
        author={Zhang, Zhilu and Wang, Haolin and Liu, Ming and Wang, Ruohao and Zuo, Wangmeng and Zhang, Jiawei},
        booktitle={ICCV},
        year={2021}
    }

## 6. Acknowledgement

This repo is built upon the framework of [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), and we borrow some code from [PyNet](https://github.com/aiff22/PyNET-PyTorch), [Zoom-Learn-Zoom](https://github.com/ceciliavision/zoom-learn-zoom), [PWC-Net](https://github.com/sniklaus/pytorch-pwc) and [AdaDSR](https://github.com/csmliu/AdaDSR), thanks for their excellent work!
