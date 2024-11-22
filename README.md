# FCNet
Reference-Based Infrared Image Colorization via Feature Enhancement and Context Refinement


## Prerequisites
- python 3.7
- torch 1.13.1
- torchvision 0.14.1
- dominate
- visdom

## Dataset
We provide the [KAIST](https://github.com/SoonminHwang/rgbt-ped-detection) dataset and the [FLIR](https://www.flir.com/oem/adas/adas-dataset-form) dataset link.


## Pretrianed
We provide the pretrained model: [Google](https://drive.google.com/drive/folders/1SW3y-mC6Sib1bZ0xPqd3GpQSeEmtjTIZ)

## Setup
```
config.yml 
  EPOCH
  BATCH_SIZE
  G_LR
  D_LR
  LAMBDA_G_FAKE
  LAMBDA_G_RECON
  LAMBDA_G_SYTLE
  LAMBDA_G_PERCEP
  LAMBDA_D_FAKE
  LAMBDA_D_REAL
  TRAIN_DIR
```

```
dataloder.py
  self.img_dir = '/home/cust/cust_data/cy/dataset/RefDayDataset/KAIST/testA'
  self.skt_dir = '/home/cust/cust_data/cy/dataset/RefDayDataset/KAIST/testB'
  self.data_list = glob.glob(os.path.join(self.img_dir, '*.jpg'))
```
Note：Modify according to your needs

## Trian
```
python main.py
```

## Test
```
python test_index.py
```

## Colorization results
### KAIST dataset
![KAIST](img/KAIST.png)


### FLIR dataset
![FLIR](img/FLIR.png)

## Acknowledgments
This code heavily borrowes from [SGA](https://github.com/kunkun0w0/SGA).

