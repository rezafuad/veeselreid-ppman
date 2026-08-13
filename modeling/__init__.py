# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .baseline import Baseline, BaselineV2, BaselineV3, BaselineV4, BaselineV5
from .baseline import StatPool, StatPoolWAtt

from .parallel import ParallelV1, ParallelV2, ParallelV3
from .parallel import ParallelV4                    # default gc-ge-mha
from .parallel import ParallelV4_gcmhage            # gc-mha-ge
from .parallel import ParallelV4_gegcmha            # ge-gc-mha
from .parallel import ParallelV4_gemhagc            # ge-mha-gc
from .parallel import ParallelV4_mhagcge            # mha-gc-ge
from .parallel import ParallelV4_mhagegc            # mha-ge-gc
from .parallel import ParallelV5 # weighted for each attention

from .parallel_feat import ParallelFeatV1, ParallelFeatV2, ParallelFeatV3, ParallelFeatV4
from .parallel_feat import ParallelFeatV5, ParallelFeatV6
from .parallel_feat import ParallelFeatV5Att

def build_model(cfg, num_classes, att_num_classes=None):
    model_split = cfg.MODEL.NAME.split("_")

    if model_split[0] == "parallelV1":
        model = ParallelV1(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT,
                    cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE, cfg.INPUT.MODALITY, cfg.MODEL.NUM_ATT_LAYERS)
    elif model_split[0] == "parallelV2":
        model = ParallelV2(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT,
                    cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE, cfg.INPUT.MODALITY, cfg.MODEL.NUM_ATT_LAYERS)
    elif model_split[0] == "parallelV3":
        model = ParallelV3(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT,
                    cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE, cfg.INPUT.MODALITY, cfg.MODEL.NUM_ATT_LAYERS)
    elif model_split[0] == "parallelV4":
        model = ParallelV4(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT,
                    cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE, cfg.INPUT.MODALITY, cfg.MODEL.NUM_ATT_LAYERS)
    elif model_split[0] == "parallelV4gcmhage":
        model = ParallelV4_gcmhage(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT,
                    cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE, cfg.INPUT.MODALITY, cfg.MODEL.NUM_ATT_LAYERS)
    elif model_split[0] == "parallelV4gegcmha":
        model = ParallelV4_gegcmha(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT,
                    cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE, cfg.INPUT.MODALITY, cfg.MODEL.NUM_ATT_LAYERS)
    elif model_split[0] == "parallelV4gemhagc":
        model = ParallelV4_gemhagc(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT,
                    cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE, cfg.INPUT.MODALITY, cfg.MODEL.NUM_ATT_LAYERS)
    elif model_split[0] == "parallelV4mhagcge":
        model = ParallelV4_mhagcge(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT,
                    cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE, cfg.INPUT.MODALITY, cfg.MODEL.NUM_ATT_LAYERS)
    elif model_split[0] == "parallelV4mhagegc":
        model = ParallelV4_mhagegc(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT,
                    cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE, cfg.INPUT.MODALITY, cfg.MODEL.NUM_ATT_LAYERS)
 
    elif model_split[0] == "parallelV5":
        model = ParallelV5(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT,
                    cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE, cfg.INPUT.MODALITY, cfg.MODEL.NUM_ATT_LAYERS)
    elif model_split[0] == "parallelfeatV1":
        model = ParallelFeatV1(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT,
                    cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE, cfg.INPUT.MODALITY, cfg.MODEL.NUM_ATT_LAYERS)
    elif model_split[0] == "parallelfeatV2":
        model = ParallelFeatV2(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT,
                    cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE, cfg.INPUT.MODALITY, cfg.MODEL.NUM_ATT_LAYERS)
    elif model_split[0] == "parallelfeatV3":
        model = ParallelFeatV3(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT,
                    cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE, cfg.INPUT.MODALITY, cfg.MODEL.NUM_ATT_LAYERS)
    elif model_split[0] == "parallelfeatV4":
        model = ParallelFeatV4(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT,
                    cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE, cfg.INPUT.MODALITY, cfg.MODEL.NUM_ATT_LAYERS)
    elif model_split[0] in ["parallelfeatV5", "parallelfeatV5gcmhage", "parallelfeatV5gegcmha", "parallelfeatV5gemhagc", "parallelfeatV5mhagcge", "parallelfeatV5mhagegc"]:
        model = ParallelFeatV5(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT,
                    cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE, cfg.INPUT.MODALITY, cfg.MODEL.NUM_ATT_LAYERS, cfg.MODEL.SACONF)
    elif model_split[0] == "parallelfeatV5att":
        model = ParallelFeatV5Att(num_classes, att_num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT,
                    cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE, cfg.INPUT.MODALITY, cfg.MODEL.NUM_ATT_LAYERS, cfg.MODEL.SACONF)
    elif model_split[0] in ["parallelfeatV6", "parallelfeatV6gcmhage", "parallelfeatV6gegcmha", "parallelfeatV6gemhagc", "parallelfeatV6mhagcge", "parallelfeatV6mhagegc"]:
        model = ParallelFeatV6(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT,
                    cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE, cfg.INPUT.MODALITY, cfg.MODEL.NUM_ATT_LAYERS, cfg.MODEL.SACONF, cfg.MODEL.PPDIV)
    elif model_split[0] == "stp":
        model = StatPool(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT,
                    cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE, cfg.INPUT.MODALITY)
    elif model_split[0] == "baselineV5":
        model = BaselineV5(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT,
                    cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE, cfg.INPUT.MODALITY)
    elif model_split[0] == "baselineV4":
        model = BaselineV4(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT,
                    cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE, cfg.INPUT.MODALITY)
    elif model_split[0] == "baselineV3":
        model = BaselineV3(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT,
                    cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE, cfg.INPUT.MODALITY)
    elif model_split[0] == "baselineV2":
        model = BaselineV2(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT,
                    cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE, cfg.INPUT.MODALITY, cfg.MODEL.NUM_ATT_LAYERS)
    else:
        model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, 
                     cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE, cfg.INPUT.MODALITY)
    return model
