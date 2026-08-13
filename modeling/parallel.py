# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import pdb
import torch
from torch import nn

from .backbones.resnet import ResNet, BasicBlock, Bottleneck 
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck 
from .backbones.squeezenet import SqueezeNet,Fire
from .backbones.densenet import _DenseLayer, _DenseBlock, _Transition, DenseNet
from .backbones.mobilenet import ConvBNReLU, InvertedResidual, MobileNetV2
from .backbones.inception import Inception3, BasicConv2d

import timm

import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.nn.functional as F

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        nn.init.constant_(m.bias, 0.0)

def get_base_model(model_name):
    base = None 
    in_planes = 288

    # MambaOut Classifier
    if model_name == 'mambaout_base':
        base = timm.create_model('mambaout_base.in1k', pretrained=True) 
        base.head = nn.Identity()
        in_planes = 768
    elif model_name == 'mambaout_small':
        base = timm.create_model('mambaout_small.in1k', pretrained=True) 
        base.head = nn.Identity()
        in_planes = 576
    elif model_name == 'mambaout_tiny':
        base = timm.create_model('mambaout_tiny.in1k', pretrained=True) 
        base.head = nn.Identity()
        in_planes = 576
    elif model_name == 'mambaout_kobe':
        base = timm.create_model('mambaout_kobe.in1k', pretrained=True) 
        base.head = nn.Identity()
        in_planes = 288
    elif model_name == 'mambaout_femto':
        base = timm.create_model('mambaout_femto.in1k', pretrained=True) 
        base.head = nn.Identity()
        in_planes = 288
    elif model_name == 'mambaout_base_plus':
        base = timm.create_model('mambaout_base_plus_rw.sw_e150_r384_in12k_ft_in1k', pretrained=True)
        base.head = nn.Identity()
        in_planes = 768

    # SwinTransformer
    if model_name == "swinv2_transformer":
        base = timm.create_model("swinv2_base_window12to16_192to256.ms_in22k_ft_in1k", pretrained=True)
        base.head = nn.Identity()
        in_planes = 1024
    elif model_name == "swinv2large_transformer":
        base = timm.create_model("swinv2_large_window12to16_192to256.ms_in22k_ft_in1k", pretrained=True)
        base.head = nn.Identity()
        in_planes = 1536

    # ViTamin
    if model_name == "vitamin_base":
        base = timm.create_model("vitamin_base_224", pretrained=True)
        base.head = nn.Identity()
        in_planes = 768

    # NaFlex
    if model_name == "naflex_base":
        base = timm.create_model("naflexvit_base_patch16_gap.e300_s576_in1k", pretrained=True)
        base.head = nn.Identity()
        in_planes = 768

    if model_name == "vit_so150m2":
        base = timm.create_model("vit_so150m2_patch16_reg1_gap_384.sbb_e200_in12k_ft_in1k", pretrained=True)
        base.head = nn.Identity()
        in_planes = 832

    return base, in_planes






class StatPool(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choicei, modality):
        super(StatPool, self).__init__()

        self.base, self.in_planes = get_base_model("_".join(model_name.split("_")[1::]))
        self.model_name = model_name

        self.conv1 = nn.Conv2d(self.in_planes, num_classes, kernel_size=1, bias=False)
        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat
        self.modality = modality

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes*3, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes*2)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes*3, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)
        print(self)

    def forward(self, x):
       
        x = self.base.forward_features(x)
        #pdb.set_trace()

        # simple statistic pooling
        if len(x.shape) > 3:
            x = torch.permute( x, (0, 3, 1, 2) )
            mean = torch.mean( x, [2,3] )
            #pdb.set_trace()
            std = torch.std( x, [2,3] )
            global_feat1 = self.gap( x )
            global_feat1 = global_feat1.view(global_feat1.shape[0], -1)
        elif len(x.shape) == 3:
            x = x[:, self.base.num_prefix_tokens:]
            x = torch.permute( x, (0, 2, 1) )
            mean = torch.mean( x, 2 )
            std = torch.std( x, 2 )
            global_feat1 = self.gap( x.view(x.shape[0], x.shape[1], x.shape[2], 1 ) )
            global_feat1 = global_feat1.view(global_feat1.shape[0], -1)
        global_feat = torch.cat( (global_feat1, mean, std), 1 )

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            #feat = self.bottleneck(global_feat)  # normalize for angular softmax
             feat = global_feat
        #pdb.set_trace()

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)['model']
        #pdb.set_trace()
        for i in param_dict:
            #print(i)
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
        ###


"""
Attention Code taken from https://github.com/KrishnaDN/Attentive-Statistics-Pooling-for-Deep-Speaker-Embedding/blob/master/modules/Attention_Pooling.py
"""
class Classic_Attention(nn.Module):
    def __init__(self,input_dim, embed_dim, attn_dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.lin_proj = nn.Linear(input_dim,embed_dim)
        self.v = torch.nn.Parameter(torch.randn(embed_dim))
                                                        
    def forward(self,inputs):
        pdb.set_trace()
        lin_out = self.lin_proj(inputs)
        v_view = self.v.unsqueeze(0).expand(lin_out.size(0), len(self.v)).unsqueeze(2)
        attention_weights = F.tanh(lin_out.bmm(v_view).squeeze())
        attention_weights_normalized = F.softmax(attention_weights,1)
        return attention_weights_normalized

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, linear=512, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear>0:
            add_block += [nn.Linear(input_dim, linear)]
        else:
            linear = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(linear)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(linear, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return [x,f]
        else:
            x = self.classifier(x)
            return x
##################################




#################################################################################################################################

from timm.models._efficientnet_blocks import SqueezeExcite, ConvBnAct, DepthwiseSeparableConv
from timm.models._efficientnet_blocks import MobileAttention
from timm.layers.global_context import GlobalContext
from timm.layers.gather_excite import GatherExcite 

"""
Att net design

    ----------- ----------- -----------
    |         | |         | |         |
x ----> att1 -+---> att1 -+---> att1 -+--> Concat ---> Final Layer 
    |---------- ----------- -----------      |
    |         | |         | |         |      |
    |-> att2 -+---> att2 -+---> att2 -+------|
    |-> att3 -+---> att3 -+---> att3 -+------|
    |         | |         | |         |      |
    ----------- ----------- -----------      |
    |                                        |
    ------------------------------------------
"""
class ParallelV1(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choicei, modality, num_attns=3):
        super(ParallelV1, self).__init__()

        self.base, self.in_planes = get_base_model("_".join(model_name.split("_")[1::]))
        self.model_name = model_name

        self.num_attns = num_attns
        self.attns1 = nn.ModuleList()
        for i in range(self.num_attns):
            self.attns1.append(GatherExcite(1024))
        self.attns2 = nn.ModuleList()
        for i in range(self.num_attns):
            self.attns2.append(GlobalContext(1024))
        self.attns3 = nn.ModuleList()
        for i in range(self.num_attns):
            self.attns3.append(MobileAttention(1024,1024))       

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat
        self.modality = modality

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes*4, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes*4)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes*4, self.num_classes)#, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):
       
        x = self.base(x)
        x = torch.permute(x, (0,3,1,2))
        #pdb.set_trace()

        x1 = x2 = x3 = x
        #pdb.set_trace()
        for att in self.attns1:
            x1 = x1 + att(x1)
        for att in self.attns2:
            x2 = x2 + att(x2)
        for att in self.attns3:
            x3 = x3 + att(x3)
        feat = torch.cat( ( self.gap(x).view(x.shape[0], x.shape[1]), 
                            self.gap(x1).view(x.shape[0], x.shape[1]),
                            self.gap(x2).view(x.shape[0], x.shape[1]),
                            self.gap(x3).view(x.shape[0], x.shape[1]),
                           ), dim=1 )

        #if self.neck == 'no':
        #    feat = torch.cat( (global_feat, self.gap(x).view(x.shape[0], -1)), dim=1 )
        #el
        #if self.neck == 'bnneck':
        #feat = self.bottleneck(feat)  # normalize for angular softmax
        #    #feat = torch.cat( (feat, self.gap(x).view(x.shape[0], -1)), dim=1 )
            
        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)['model']
        #pdb.set_trace()
        for i in param_dict:
            #print(i)
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
        ###





"""
Att net design


"""
class ParallelV2(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choicei, modality, num_attns=3):
        super(ParallelV2, self).__init__()

        self.base, self.in_planes = get_base_model("_".join(model_name.split("_")[1::]))
        self.model_name = model_name

        self.num_attns = num_attns
        self.attns1 = nn.ModuleList()
        for i in range(self.num_attns):
            if i % 3 == 2:
                self.attns1.append(MobileAttention(1024,1024))       
            elif i % 3 == 1:
                self.attns1.append(GatherExcite(1024))
            else:
                self.attns1.append(GlobalContext(1024))
        self.attns2 = nn.ModuleList()
        for i in range(self.num_attns):
            if i % 3 == 2:
                self.attns2.append(MobileAttention(1024,1024))       
            elif i % 3 == 1:
                self.attns2.append(GatherExcite(1024))
            else:
                self.attns2.append(GlobalContext(1024)) 
        self.attns3 = nn.ModuleList()
        for i in range(self.num_attns):
            if i % 3 == 2:
                self.attns3.append(MobileAttention(1024,1024))       
            elif i % 3 == 1:
                self.attns3.append(GatherExcite(1024))
            else:
                self.attns3.append(GlobalContext(1024))
            ####
        ####
 
        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat
        self.modality = modality

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes*4, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes*4)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes*4, self.num_classes)#, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):
       
        x = self.base(x)
        x = torch.permute(x, (0,3,1,2))
        #pdb.set_trace()

        x1 = x2 = x3 = x
        #pdb.set_trace()
        for att in self.attns1:
            x1 = x1 + att(x1)
        for att in self.attns2:
            x2 = x2 + att(x2)
        for att in self.attns3:
            x3 = x3 + att(x3)
        feat = torch.cat( ( self.gap(x).view(x.shape[0], x.shape[1]), 
                            self.gap(x1).view(x.shape[0], x.shape[1]),
                            self.gap(x2).view(x.shape[0], x.shape[1]),
                            self.gap(x3).view(x.shape[0], x.shape[1]),
                           ), dim=1 )


        #if self.neck == 'no':
        #    feat = torch.cat( (global_feat, self.gap(x).view(x.shape[0], -1)), dim=1 )
        #el
        #if self.neck == 'bnneck':
        #feat = self.bottleneck(feat)  # normalize for angular softmax
        #    #feat = torch.cat( (feat, self.gap(x).view(x.shape[0], -1)), dim=1 )
            
        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)['model']
        #pdb.set_trace()
        for i in param_dict:
            #print(i)
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
        ###




######
## With Dropout2D
class ParallelV3(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choicei, modality, num_attns=3):
        super(ParallelV3, self).__init__()

        self.base, self.in_planes = get_base_model("_".join(model_name.split("_")[1::]))
        self.model_name = model_name

        self.num_attns = num_attns
        self.attns1 = nn.ModuleList()
        for i in range(self.num_attns):
            self.attns1.append(GatherExcite(1024))
        self.attns2 = nn.ModuleList()
        for i in range(self.num_attns):
            self.attns2.append(GlobalContext(1024))
        self.attns3 = nn.ModuleList()
        for i in range(self.num_attns):
            self.attns3.append(MobileAttention(1024,1024))       

        self.dp2d1 = nn.Dropout2d(0.035)
        self.dp2d2 = nn.Dropout2d(0.035)
        self.dp2d3 = nn.Dropout2d(0.035)
        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat
        self.modality = modality

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes*4, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes*4)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes*4, self.num_classes)#, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):
       
        x = self.base(x)
        x = torch.permute(x, (0,3,1,2))
        #pdb.set_trace()

        x1 = x2 = x3 = x
        #pdb.set_trace()
        for att in self.attns1:
            x1 = x1 + att(x1)
        for att in self.attns2:
            x2 = x2 + att(x2)
        for att in self.attns3:
            x3 = x3 + att(x3)
        x1 = self.dp2d1(x1)
        x2 = self.dp2d2(x2)
        x3 = self.dp2d3(x3)
        feat = torch.cat( ( self.gap(x).view(x.shape[0], x.shape[1]), 
                            self.gap(x1).view(x.shape[0], x.shape[1]),
                            self.gap(x2).view(x.shape[0], x.shape[1]),
                            self.gap(x3).view(x.shape[0], x.shape[1]),
                           ), dim=1 )

        #if self.neck == 'no':
        #    feat = torch.cat( (global_feat, self.gap(x).view(x.shape[0], -1)), dim=1 )
        #el
        #if self.neck == 'bnneck':
        #feat = self.bottleneck(feat)  # normalize for angular softmax
        #    #feat = torch.cat( (feat, self.gap(x).view(x.shape[0], -1)), dim=1 )
            
        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)['model']
        #pdb.set_trace()
        for i in param_dict:
            #print(i)
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
        ###


####
class ParallelV4(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choicei, modality, num_attns=3):
        super(ParallelV4, self).__init__()

        self.base, self.in_planes = get_base_model("_".join(model_name.split("_")[1::]))
        self.model_name = model_name

        self.num_attns = num_attns
        self.attns1 = nn.ModuleList()
        for i in range(self.num_attns):
            if i % 3 == 2:
                self.attns1.append(MobileAttention(1024,1024))       
            elif i % 3 == 1:
                self.attns1.append(GatherExcite(1024))
            else:
                self.attns1.append(GlobalContext(1024))
        self.attns2 = nn.ModuleList()
        for i in range(self.num_attns):
            if i % 3 == 2:
                self.attns2.append(MobileAttention(1024,1024))       
            elif i % 3 == 1:
                self.attns2.append(GatherExcite(1024))
            else:
                self.attns2.append(GlobalContext(1024)) 
        self.attns3 = nn.ModuleList()
        for i in range(self.num_attns):
            if i % 3 == 2:
                self.attns3.append(MobileAttention(1024,1024))       
            elif i % 3 == 1:
                self.attns3.append(GatherExcite(1024))
            else:
                self.attns3.append(GlobalContext(1024))
            ####
        ####

        self.dp2d1 = nn.Dropout2d(0.03125)
        self.dp2d2 = nn.Dropout2d(0.03125)
        self.dp2d3 = nn.Dropout2d(0.03125)
 
 
        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat
        self.modality = modality

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes*4, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes*4)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes*4, self.num_classes)#, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):
       
        x = self.base(x)
        x = torch.permute(x, (0,3,1,2))
        #pdb.set_trace()

        x1 = x2 = x3 = x
        #pdb.set_trace()
        for att in self.attns1:
            x1 = x1 + att(x1)
        for att in self.attns2:
            x2 = x2 + att(x2)
        for att in self.attns3:
            x3 = x3 + att(x3)
        x1 = self.dp2d1(x1)
        x2 = self.dp2d2(x2)
        x3 = self.dp2d3(x3)
        feat = torch.cat( ( self.gap(x).view(x.shape[0], x.shape[1]), 
                            self.gap(x1).view(x.shape[0], x.shape[1]),
                            self.gap(x2).view(x.shape[0], x.shape[1]),
                            self.gap(x3).view(x.shape[0], x.shape[1]),
                           ), dim=1 )


        #if self.neck == 'no':
        #    feat = torch.cat( (global_feat, self.gap(x).view(x.shape[0], -1)), dim=1 )
        #el
        #if self.neck == 'bnneck':
        #feat = self.bottleneck(feat)  # normalize for angular softmax
        #    #feat = torch.cat( (feat, self.gap(x).view(x.shape[0], -1)), dim=1 )
            
        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)['model']
        #pdb.set_trace()
        for i in param_dict:
            #print(i)
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
        ###

class ParallelV4_gcmhage(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choicei, modality, num_attns=3):
        super(ParallelV4_gcmhage, self).__init__()

        #pdb.set_trace()
        self.base, self.in_planes = get_base_model("_".join(model_name.split("_")[1::]))
        self.model_name = model_name

        self.num_attns = num_attns
        self.attns1 = nn.ModuleList()
        for i in range(self.num_attns):
            if i % 3 == 1:
                self.attns1.append(MobileAttention(1024,1024))       
            elif i % 3 == 2:
                self.attns1.append(GatherExcite(1024))
            else:
                self.attns1.append(GlobalContext(1024))
        self.attns2 = nn.ModuleList()
        for i in range(self.num_attns):
            if i % 3 == 1:
                self.attns2.append(MobileAttention(1024,1024))       
            elif i % 3 == 2:
                self.attns2.append(GatherExcite(1024))
            else:
                self.attns2.append(GlobalContext(1024)) 
        self.attns3 = nn.ModuleList()
        for i in range(self.num_attns):
            if i % 3 == 1:
                self.attns3.append(MobileAttention(1024,1024))       
            elif i % 3 == 2:
                self.attns3.append(GatherExcite(1024))
            else:
                self.attns3.append(GlobalContext(1024))
            ####
        ####

        self.dp2d1 = nn.Dropout2d(0.03125)
        self.dp2d2 = nn.Dropout2d(0.03125)
        self.dp2d3 = nn.Dropout2d(0.03125)
 
 
        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat
        self.modality = modality

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes*4, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes*4)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes*4, self.num_classes)#, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):
       
        x = self.base(x)
        x = torch.permute(x, (0,3,1,2))
        #pdb.set_trace()

        x1 = x2 = x3 = x
        #pdb.set_trace()
        for att in self.attns1:
            x1 = x1 + att(x1)
        for att in self.attns2:
            x2 = x2 + att(x2)
        for att in self.attns3:
            x3 = x3 + att(x3)
        x1 = self.dp2d1(x1)
        x2 = self.dp2d2(x2)
        x3 = self.dp2d3(x3)
        feat = torch.cat( ( self.gap(x).view(x.shape[0], x.shape[1]), 
                            self.gap(x1).view(x.shape[0], x.shape[1]),
                            self.gap(x2).view(x.shape[0], x.shape[1]),
                            self.gap(x3).view(x.shape[0], x.shape[1]),
                           ), dim=1 )


        #if self.neck == 'no':
        #    feat = torch.cat( (global_feat, self.gap(x).view(x.shape[0], -1)), dim=1 )
        #el
        #if self.neck == 'bnneck':
        #feat = self.bottleneck(feat)  # normalize for angular softmax
        #    #feat = torch.cat( (feat, self.gap(x).view(x.shape[0], -1)), dim=1 )
            
        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)['model']
        #pdb.set_trace()
        for i in param_dict:
            #print(i)
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
        ###



class ParallelV4_gegcmha(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choicei, modality, num_attns=3):
        super(ParallelV4_gegcmha, self).__init__()

        #pdb.set_trace()
        self.base, self.in_planes = get_base_model("_".join(model_name.split("_")[1::]))
        self.model_name = model_name

        self.num_attns = num_attns
        self.attns1 = nn.ModuleList()
        for i in range(self.num_attns):
            if i % 3 == 2:
                self.attns1.append(MobileAttention(1024,1024))       
            elif i % 3 == 0:
                self.attns1.append(GatherExcite(1024))
            else:
                self.attns1.append(GlobalContext(1024))
        self.attns2 = nn.ModuleList()
        for i in range(self.num_attns):
            if i % 3 == 2:
                self.attns2.append(MobileAttention(1024,1024))       
            elif i % 3 == 0:
                self.attns2.append(GatherExcite(1024))
            else:
                self.attns2.append(GlobalContext(1024)) 
        self.attns3 = nn.ModuleList()
        for i in range(self.num_attns):
            if i % 3 == 2:
                self.attns3.append(MobileAttention(1024,1024))       
            elif i % 3 == 0:
                self.attns3.append(GatherExcite(1024))
            else:
                self.attns3.append(GlobalContext(1024))
            ####
        ####

        self.dp2d1 = nn.Dropout2d(0.03125)
        self.dp2d2 = nn.Dropout2d(0.03125)
        self.dp2d3 = nn.Dropout2d(0.03125)
 
 
        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat
        self.modality = modality

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes*4, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes*4)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes*4, self.num_classes)#, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):
       
        x = self.base(x)
        x = torch.permute(x, (0,3,1,2))
        #pdb.set_trace()

        x1 = x2 = x3 = x
        #pdb.set_trace()
        for att in self.attns1:
            x1 = x1 + att(x1)
        for att in self.attns2:
            x2 = x2 + att(x2)
        for att in self.attns3:
            x3 = x3 + att(x3)
        x1 = self.dp2d1(x1)
        x2 = self.dp2d2(x2)
        x3 = self.dp2d3(x3)
        feat = torch.cat( ( self.gap(x).view(x.shape[0], x.shape[1]), 
                            self.gap(x1).view(x.shape[0], x.shape[1]),
                            self.gap(x2).view(x.shape[0], x.shape[1]),
                            self.gap(x3).view(x.shape[0], x.shape[1]),
                           ), dim=1 )


        #if self.neck == 'no':
        #    feat = torch.cat( (global_feat, self.gap(x).view(x.shape[0], -1)), dim=1 )
        #el
        #if self.neck == 'bnneck':
        #feat = self.bottleneck(feat)  # normalize for angular softmax
        #    #feat = torch.cat( (feat, self.gap(x).view(x.shape[0], -1)), dim=1 )
            
        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)['model']
        #pdb.set_trace()
        for i in param_dict:
            #print(i)
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
        ###


class ParallelV4_gemhagc(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choicei, modality, num_attns=3):
        super(ParallelV4_gemhagc, self).__init__()

        #pdb.set_trace()
        self.base, self.in_planes = get_base_model("_".join(model_name.split("_")[1::]))
        self.model_name = model_name

        self.num_attns = num_attns
        self.attns1 = nn.ModuleList()
        for i in range(self.num_attns):
            if i % 3 == 1:
                self.attns1.append(MobileAttention(1024,1024))       
            elif i % 3 == 0:
                self.attns1.append(GatherExcite(1024))
            else:
                self.attns1.append(GlobalContext(1024))
        self.attns2 = nn.ModuleList()
        for i in range(self.num_attns):
            if i % 3 == 1:
                self.attns2.append(MobileAttention(1024,1024))       
            elif i % 3 == 0:
                self.attns2.append(GatherExcite(1024))
            else:
                self.attns2.append(GlobalContext(1024)) 
        self.attns3 = nn.ModuleList()
        for i in range(self.num_attns):
            if i % 3 == 1:
                self.attns3.append(MobileAttention(1024,1024))       
            elif i % 3 == 0:
                self.attns3.append(GatherExcite(1024))
            else:
                self.attns3.append(GlobalContext(1024))
            ####
        ####

        self.dp2d1 = nn.Dropout2d(0.03125)
        self.dp2d2 = nn.Dropout2d(0.03125)
        self.dp2d3 = nn.Dropout2d(0.03125)
 
 
        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat
        self.modality = modality

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes*4, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes*4)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes*4, self.num_classes)#, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):
       
        x = self.base(x)
        x = torch.permute(x, (0,3,1,2))
        #pdb.set_trace()

        x1 = x2 = x3 = x
        #pdb.set_trace()
        for att in self.attns1:
            x1 = x1 + att(x1)
        for att in self.attns2:
            x2 = x2 + att(x2)
        for att in self.attns3:
            x3 = x3 + att(x3)
        x1 = self.dp2d1(x1)
        x2 = self.dp2d2(x2)
        x3 = self.dp2d3(x3)
        feat = torch.cat( ( self.gap(x).view(x.shape[0], x.shape[1]), 
                            self.gap(x1).view(x.shape[0], x.shape[1]),
                            self.gap(x2).view(x.shape[0], x.shape[1]),
                            self.gap(x3).view(x.shape[0], x.shape[1]),
                           ), dim=1 )


        #if self.neck == 'no':
        #    feat = torch.cat( (global_feat, self.gap(x).view(x.shape[0], -1)), dim=1 )
        #el
        #if self.neck == 'bnneck':
        #feat = self.bottleneck(feat)  # normalize for angular softmax
        #    #feat = torch.cat( (feat, self.gap(x).view(x.shape[0], -1)), dim=1 )
            
        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)['model']
        #pdb.set_trace()
        for i in param_dict:
            #print(i)
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
        ###

class ParallelV4_mhagcge(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choicei, modality, num_attns=3):
        super(ParallelV4_mhagcge, self).__init__()

        #pdb.set_trace()
        self.base, self.in_planes = get_base_model("_".join(model_name.split("_")[1::]))
        self.model_name = model_name

        self.num_attns = num_attns
        self.attns1 = nn.ModuleList()
        for i in range(self.num_attns):
            if i % 3 == 0:
                self.attns1.append(MobileAttention(1024,1024))       
            elif i % 3 == 2:
                self.attns1.append(GatherExcite(1024))
            else:
                self.attns1.append(GlobalContext(1024))
        self.attns2 = nn.ModuleList()
        for i in range(self.num_attns):
            if i % 3 == 0:
                self.attns2.append(MobileAttention(1024,1024))       
            elif i % 3 == 2:
                self.attns2.append(GatherExcite(1024))
            else:
                self.attns2.append(GlobalContext(1024)) 
        self.attns3 = nn.ModuleList()
        for i in range(self.num_attns):
            if i % 3 == 0:
                self.attns3.append(MobileAttention(1024,1024))       
            elif i % 3 == 2:
                self.attns3.append(GatherExcite(1024))
            else:
                self.attns3.append(GlobalContext(1024))
            ####
        ####

        self.dp2d1 = nn.Dropout2d(0.03125)
        self.dp2d2 = nn.Dropout2d(0.03125)
        self.dp2d3 = nn.Dropout2d(0.03125)
 
 
        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat
        self.modality = modality

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes*4, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes*4)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes*4, self.num_classes)#, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):
       
        x = self.base(x)
        x = torch.permute(x, (0,3,1,2))
        #pdb.set_trace()

        x1 = x2 = x3 = x
        #pdb.set_trace()
        for att in self.attns1:
            x1 = x1 + att(x1)
        for att in self.attns2:
            x2 = x2 + att(x2)
        for att in self.attns3:
            x3 = x3 + att(x3)
        x1 = self.dp2d1(x1)
        x2 = self.dp2d2(x2)
        x3 = self.dp2d3(x3)
        feat = torch.cat( ( self.gap(x).view(x.shape[0], x.shape[1]), 
                            self.gap(x1).view(x.shape[0], x.shape[1]),
                            self.gap(x2).view(x.shape[0], x.shape[1]),
                            self.gap(x3).view(x.shape[0], x.shape[1]),
                           ), dim=1 )


        #if self.neck == 'no':
        #    feat = torch.cat( (global_feat, self.gap(x).view(x.shape[0], -1)), dim=1 )
        #el
        #if self.neck == 'bnneck':
        #feat = self.bottleneck(feat)  # normalize for angular softmax
        #    #feat = torch.cat( (feat, self.gap(x).view(x.shape[0], -1)), dim=1 )
            
        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)['model']
        #pdb.set_trace()
        for i in param_dict:
            #print(i)
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
        ###

class ParallelV4_mhagegc(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choicei, modality, num_attns=3):
        super(ParallelV4_mhagegc, self).__init__()

        #pdb.set_trace()
        self.base, self.in_planes = get_base_model("_".join(model_name.split("_")[1::]))
        self.model_name = model_name

        self.num_attns = num_attns
        self.attns1 = nn.ModuleList()
        for i in range(self.num_attns):
            if i % 3 == 0:
                self.attns1.append(MobileAttention(1024,1024))       
            elif i % 3 == 1:
                self.attns1.append(GatherExcite(1024))
            else:
                self.attns1.append(GlobalContext(1024))
        self.attns2 = nn.ModuleList()
        for i in range(self.num_attns):
            if i % 3 == 0:
                self.attns2.append(MobileAttention(1024,1024))       
            elif i % 3 == 1:
                self.attns2.append(GatherExcite(1024))
            else:
                self.attns2.append(GlobalContext(1024)) 
        self.attns3 = nn.ModuleList()
        for i in range(self.num_attns):
            if i % 3 == 0:
                self.attns3.append(MobileAttention(1024,1024))       
            elif i % 3 == 1:
                self.attns3.append(GatherExcite(1024))
            else:
                self.attns3.append(GlobalContext(1024))
            ####
        ####

        self.dp2d1 = nn.Dropout2d(0.03125)
        self.dp2d2 = nn.Dropout2d(0.03125)
        self.dp2d3 = nn.Dropout2d(0.03125)
 
 
        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat
        self.modality = modality

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes*4, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes*4)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes*4, self.num_classes)#, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):
       
        x = self.base(x)
        x = torch.permute(x, (0,3,1,2))
        #pdb.set_trace()

        x1 = x2 = x3 = x
        #pdb.set_trace()
        for att in self.attns1:
            x1 = x1 + att(x1)
        for att in self.attns2:
            x2 = x2 + att(x2)
        for att in self.attns3:
            x3 = x3 + att(x3)
        x1 = self.dp2d1(x1)
        x2 = self.dp2d2(x2)
        x3 = self.dp2d3(x3)
        feat = torch.cat( ( self.gap(x).view(x.shape[0], x.shape[1]), 
                            self.gap(x1).view(x.shape[0], x.shape[1]),
                            self.gap(x2).view(x.shape[0], x.shape[1]),
                            self.gap(x3).view(x.shape[0], x.shape[1]),
                           ), dim=1 )


        #if self.neck == 'no':
        #    feat = torch.cat( (global_feat, self.gap(x).view(x.shape[0], -1)), dim=1 )
        #el
        #if self.neck == 'bnneck':
        #feat = self.bottleneck(feat)  # normalize for angular softmax
        #    #feat = torch.cat( (feat, self.gap(x).view(x.shape[0], -1)), dim=1 )
            
        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)['model']
        #pdb.set_trace()
        for i in param_dict:
            #print(i)
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
        ###






######
## With Dropout2D + parallel weighted network
class ParallelV5(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choicei, modality, num_attns=3):
        super(ParallelV5, self).__init__()

        self.base, self.in_planes = get_base_model("_".join(model_name.split("_")[1::]))
        self.model_name = model_name

        self.num_attns = num_attns
        self.attns1 = nn.ModuleList()
        for i in range(self.num_attns):
            self.attns1.append(GatherExcite(1024))
        self.attns2 = nn.ModuleList()
        for i in range(self.num_attns):
            self.attns2.append(GlobalContext(1024))
        self.attns3 = nn.ModuleList()
        for i in range(self.num_attns):
            self.attns3.append(MobileAttention(1024,1024))       

        self.dp2d1 = nn.Dropout2d(0.035)
        self.dp2d2 = nn.Dropout2d(0.035)
        self.dp2d3 = nn.Dropout2d(0.035)

        ####
        self.bottleneckConv = nn.Sequential(
                    nn.Conv2d(1024*3, 1024*3, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(),
                    nn.BatchNorm2d(1024*3),
                    nn.Conv2d(1024*3, 1024*3, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(),
                    nn.BatchNorm2d(1024*3),
                    #nn.AdaptiveAvgPool2d((1,1)),
                )
        self.bottleneckFC = nn.Sequential(
                    nn.Conv1d(3, 3, 1, 1, 0),
                    nn.ReLU(),
                    nn.Conv1d(3, 1, 1, 1, 0),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.BatchNorm1d(512)
                ) 
        ####


        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat
        self.modality = modality

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes*4, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes*4)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes*4, self.num_classes)#, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):
       
        x = self.base(x)
        x = torch.permute(x, (0,3,1,2))
        #pdb.set_trace()

        x1 = x2 = x3 = x
        #pdb.set_trace()

        #pdb.set_trace()
        for i in range(self.num_attns):
            x1 = x1 + self.attns1[i](x1)
            x2 = x2 + self.attns2[i](x2)
            x3 = x3 + self.attns3[i](x3)
            x_weighted = self.bottleneckConv(torch.cat( (x1, x2, x3), dim=1 ))
            x1 = x_weighted[:, 0:1024, :, :]
            x2 = x_weighted[:, 1024:2048, :, :]
            x3 = x_weighted[:, 2048:3072, :, :]
        ####

        x1 = self.dp2d1(x1)
        x2 = self.dp2d2(x2)
        x3 = self.dp2d3(x3)

        #pdb.set_trace()
        feat = torch.cat( ( self.gap(x).view(x.shape[0], x.shape[1]), 
                            self.gap(x_weighted).view(x_weighted.shape[0], x_weighted.shape[1])
                           ), dim=1 )

        #if self.neck == 'no':
        #    feat = torch.cat( (global_feat, self.gap(x).view(x.shape[0], -1)), dim=1 )
        #el
        #if self.neck == 'bnneck':
        #feat = self.bottleneck(feat)  # normalize for angular softmax
        #    #feat = torch.cat( (feat, self.gap(x).view(x.shape[0], -1)), dim=1 )
            
        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)['model']
        #pdb.set_trace()
        for i in param_dict:
            #print(i)
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
        ###


####

