# PROTOSEG

Code for our paper: Unified 3D Segmenter As Prototypical Classifiers

![arch](img/arch.png)

## Data Preparation

### ScanNet v2

The preprocessing support semantic and instance segmentation for both `ScanNet20` and `ScanNet200`.

- Download the [ScanNet](http://www.scan-net.org/) v2 dataset.
- Run preprocessing code for raw ScanNet as follows:

```bash
# RAW_SCANNET_DIR: the directory of downloaded ScanNet v2 raw dataset.
# PROCESSED_SCANNET_DIR: the directory of processed ScanNet dataset (output dir).
python pcr/datasets/preprocessing/scannet/preprocess_scannet.py --dataset_root ${RAW_SCANNET_DIR} --output_root ${PROCESSED_SCANNET_DIR}
```

- Link processed dataset to codebase:
```bash
# PROCESSED_SCANNET_DIR: the directory of processed ScanNet dataset.
mkdir data
ln -s ${RAW_SCANNET_DIR} ${CODEBASE_DIR}/data/scannet
```

### S3DIS

- Download S3DIS data by filling this [Google form](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1). Download the `Stanford3dDataset_v1.2_Aligned_Version.zip` file and unzip it.
- The original S3DIS data contains some bugs data need manually fix it. `xxx^@xxx`
- Run preprocessing code for S3DIS as follows:

```bash
# RAW_S3DIS_DIR: the directory of downloaded Stanford3dDataset_v1.2_Aligned_Version dataset.
# PROCESSED_S3DIS_DIR: the directory of processed s3dis dataset (output dir).
python pcr/datasets/preprocessing/s3dis/preprocess_s3dis.py --dataset_root ${RAW_S3DIS_DIR} --output_root ${PROCESSED_S3DIS_DIR}
```
- Link processed dataset to codebase.
```bash
# PROCESSED_S3DIS_DIR: the directory of processed s3dis dataset.
mkdir data
ln -s ${RAW_S3DIS_DIR} ${CODEBASE_DIR}/data/s3dis
```

### Semantic KITTI
- Download [Semantic KITTI](http://www.semantic-kitti.org/dataset.html#download) dataset.
- Link dataset to codebase.
```bash
# SEMANTIC_KITTI_DIR: the directory of Semantic KITTI dataset.
mkdir data
ln -s ${SEMANTIC_KITTI_DIR} ${CODEBASE_DIR}/data/semantic_kitti
```

## Training
```bash
# ScanNet
sh scripts/train.sh -g 4 -d scannet -c semseg-protoseg-0-base -n semseg-protoseg-0-base
# s3dis
sh scripts/train.sh -g 4 -d s3dis -c semseg-protoseg-0-base -n semseg-protoseg-0-base
```

# Acknowledge
```shell
@inproceedings{wu2022point,
  title={Point transformer v2: Grouped vector attention and partition-based pooling},
  author={Wu, Xiaoyang and Lao, Yixing and Jiang, Li and Liu, Xihui and Zhao, Hengshuang},
  booktitle={NeurIPS},
  year={2022}
}
```

```shell
@inproceedings{wang2023visual,
  title={Visual recognition with deep nearest centroids},
  author={Wang, Wenguan and Han, Cheng and Zhou, Tianfei and Liu, Dongfang},
  booktitle={ICLR},
  year={2023}
}
```
