# Part pooling with mixture of attention networks for image-based vessel reidentification

Reza Fuad Rachmadi, Anggit Wikanningrum, Khusnul Muchlisin, I Ketut Eddy Purnama

Paper Link: https://doi.org/10.1016/j.ijcce.2026.05.006

## Abstract

In vessel traffic management systems, an Automatic Identification System (AIS) is usually used to track vessel positions and control 
the flow. One disadvantage of AIS is that the ship's crew can deactivate the transceiver installed on the ship. To support the AIS, 
other systems need to be implemented, including image-based vessel reidentification systems. In this paper, we propose a part pooling 
with a mixture of attention networks (PP-MAN) model for image-based vessel reidentification problems. The proposed model is formed by 
attaching mixtures of attention networks to a Swin Transformer V2 backbone with a part-pooling mechanism. Three different attention 
mechanisms were used to form the MAN module, including Multi-Head Attention, Gather Excite, and Global Context. To test the performance 
of our proposed model, we used three reidentification datasets, including Warship, VesselReID-1248, and ShipReID-2400. Experiments on 
those three datasets show that our proposed model achieves state-of-the-art performance across all datasets, with an mAP of 89.1% and 
rank-1 of 97.2% on the Warship dataset, an mAP of 61.3% and rank-1 of 72.3% on the VesselReID-1248 dataset, and an mAP of 42.7% and 
rank-1 of 50.3% on the ShipReID-2400 dataset. Further GradCAM and t-SNE analysis show that our proposed model extracted features from 
some vessel parts and can distinguish between vessel IDs.

## Model Architecture

![Model Architecture](./model.png)

## Comparison with SOTA

![Comparison with SOTA](./comparison-sota.png)


## Training 

To do

## Testing

Testing from pretrained weight:

```bash
python -u test.py --config_file='swinv2_softmax_triplet.yml' MODEL.DEVICE_ID "('0')" DATASETS.NAMES "market1501v2" DATASETS.ROOT_DIR "../data/ShipReID-2400/testing/" MODEL.NAME "parallelfeatV5_swinv2_transformer"  MODEL.NUM_ATT_LAYERS 3 INPUT.SIZE_TRAIN [256,256] INPUT.SIZE_TEST [256,256] TEST.WEIGHT "./shipreid2400_ParallelFeatV5_swinv2_transformer_trial2/parallelfeatV5_swinv2_transformer_checkpoint_179_avg_acc=0.9991.pt" OUTPUT_DIR .
```

Change the value of `DATASETS.ROOT_DIR` to the dataset directory. 

## Pretrained Weights

Currently, only the ShipReID-2400 dataset weight is available. We plan to release all three dataset weights in the future. 

| Dataset | Pretrained weight download link |
| ------- | ------------------------------- |
| ShipReID-2400 | [https://drive.google.com/file/d/1Lq65eeMZ-oVoOzZJ9nK_wG_44Ihn2ErF/view?usp=sharing] |

## Credits

Code is derived from [MultiModal Vehicle ReID](https://github.com/ttaalle/multi-modal-vehicle-Re-ID). The original README.md can be accessed from the README_ORIGINAL.md file.

