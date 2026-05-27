from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import itertools
import timm
import torchvision
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from typing import List, Optional, Tuple, Union

from transformers import MBartForConditionalGeneration

from sage.utils.get_default_hyperparameters import config_decoder
from sage.utils import utils

# global definition
from sage.utils.definition import *


from pathlib import Path


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


def make_visual_backbone(name='resnet18'):
    if name.startswith('resnet'):
        from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights
        if name == 'resnet18':
            model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        elif name == 'resnet34':
            model = torchvision.models.resnet34(weights=ResNet34_Weights.DEFAULT)
        elif name == 'resnet50':
            model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
        elif name == 'resnet101':
            model = torchvision.models.resnet101(weights=ResNet101_Weights.DEFAULT)
        else:
            raise ValueError(f'Unsupported resnet variant: {name}')
        inchannel = model.fc.in_features
        model.fc = nn.Identity()
    elif name == 'dinov2s':
        model = timm.create_model(
            'timm/vit_small_patch14_dinov2.lvd142m',
            img_size=224,
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
        )
        for param in model.parameters():
            param.requires_grad = False

    return model

class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()
        self.resnet = make_visual_backbone(name='resnet34')

    def forward(self, x, lengths):
        x = self.resnet(x)
        x_batch = []
        start = 0
        assert sum(lengths) == x.shape[0], 'lengths should have the same length as x'
        for length in lengths:
            end = start + length
            x_batch.append(x[start:end])
            start = end

        skipconnect = None

        x = pad_sequence(x_batch,padding_value=PAD_IDX,batch_first=True)
        return x, skipconnect
  
class TemporalConv(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type=2):
        super(TemporalConv, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.conv_type = conv_type

        if self.conv_type == 0:
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K5', "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]
        elif self.conv_type == 3:
            self.kernel_size = ['K5']
        elif self.conv_type == 4:
            self.kernel_size = ['K3', 'K5']

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            if ks[0] == 'P':
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                modules.append(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules)
    
    def forward(self, x):
        x = self.temporal_conv(x.permute(0,2,1))
        return x.permute(0,2,1)
    
def make_head(inplanes, planes, head_type):
    if head_type == 'linear':
        return nn.Linear(inplanes, planes, bias=False)
    elif head_type == 'TV_Projector':
        return nn.Sequential(
            Linear_Block(inplanes, planes),
            nn.GELU(),
            nn.Dropout(0.2),
            Linear_Block(planes, planes),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(planes, planes),
        )
    else:
        return nn.Identity()
    
class Linear_Block(nn.Module):
    def __init__(self, in_dim=1024, out_dim=2328):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight, gain=1.0)  # For GELU
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        return x

class TextCLIP(nn.Module):
    def __init__(self, config=None, inplanes=1024, planes=1024, head_type='identy', device=None, frozen=False):
        super(TextCLIP, self).__init__()

        self.model_txt = MBartForConditionalGeneration.from_pretrained(config['model']['transformer']).get_encoder() 
        self.device = device
        self.lm_head = make_head(inplanes, planes, head_type)
        if frozen:
            print('Freezing the text model')
            for param in self.model_txt.parameters():
                param.requires_grad = False

    def forward(self, tgt_input):
        
        txt_logits = self.model_txt(input_ids=tgt_input['input_ids'].to(self.device), attention_mask=tgt_input['attention_mask'].to(self.device))[0]
        last_token = tgt_input['attention_mask'].sum(dim=-1)
        assert last_token.shape[0] == txt_logits.shape[0], f'last_token {last_token.shape[0]} and txt_logits {txt_logits.shape[0]} should have the same batch size'
        output = torch.stack([logits[:l-1,:].mean(dim=0) for logits, l in zip(txt_logits, last_token)], dim=0)
        
        # output = txt_logits[torch.arange(txt_logits.shape[0]), tgt_input['input_ids'].argmax(dim=-1)]
        return self.lm_head(output), txt_logits

class Segment_Pretrain(nn.Module):
    def __init__(self, config, inplanes=1024, planes=1024, head_type='linear', device=None, frozen=False):
        super().__init__()
        self.config = config
        self.model =  FeatureExtracter() 
        self.device = device
        # self.global_encoder = MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder']).get_encoder()
        self.global_encoder = MBartForConditionalGeneration.from_pretrained(config['model']['transformer']).get_encoder()

        if frozen:
            print('Freezing the Transformer model')
            for param in self.global_encoder.parameters():
                param.requires_grad = False

        self.vis_lang_proj = make_head(inplanes, planes, 'TV_Projector')
        
    def forward(self, src_input):
        x, skipconnect = self.model(src_input['input_ids'].to(self.device), src_input['src_length_batch'].to(self.device)) # [b, n, c]
        attention_mask = src_input['attention_mask']

        mask = attention_mask.to(self.device).unsqueeze(-1).float() 
        sum_embeddings = (x * mask).sum(dim=1)
        sum_mask = mask.sum(dim=1)  # B x 1
        output_cls_rep = sum_embeddings / sum_mask  # B x C 

        output_cls_rep = self.vis_lang_proj(output_cls_rep)

        seg_ids = src_input['seg_ids']

        grouped_embeddings = [torch.stack([output_cls_rep[i] for i in indices])
                            for _, indices in itertools.groupby(range(len(seg_ids)), key=lambda i: seg_ids[i])]

        padded_embeddings = pad_sequence(grouped_embeddings, padding_value=PAD_IDX, batch_first=True)
        final_attention_mask = (padded_embeddings[:, :, 0] != PAD_IDX).long()

        global_outs = self.global_encoder(inputs_embeds=padded_embeddings, attention_mask=final_attention_mask.to(self.device), return_dict=True)
        last_hidden_state = global_outs['last_hidden_state']

        return padded_embeddings, last_hidden_state, final_attention_mask

        
class SAGEEncoder(nn.Module):
    def __init__(self, config, inplanes=1024, embed_dim=1024, device=None):
        super(SAGEEncoder, self).__init__()
        # self.model_txt = TextCLIP(config, inplanes=embed_dim, planes=embed_dim, device=device, frozen=True)
        self.device = device
        self.model_images = Segment_Pretrain(config, inplanes=inplanes, planes=embed_dim, device=device, frozen=False)

        self.logit_scale_input = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def cross_lingual_similarity(self, features1, features2, feature1_att_mask, feature2_att_mask, passed_logit_scale=None):
        # Reference for this function: https://github.com/FangyunWei/SLRT/tree/main/CiCo

        # features1: (B, T, 1024) - features2: (B, L, 1024)
        # normalize
        features1 = features1 / features1.norm(dim=-1, keepdim=True)
        features2 = features2 / features2.norm(dim=-1, keepdim=True)

        # matrix multiplication
        dot_p = torch.einsum("ais,bjs->abij", [features1, features2])  #  (B, B, T, L)
        # row-wise softmax and element-wise multiply, then row-wise sum
        after_softmax_i2t = torch.nansum(dot_p * torch.softmax(dot_p/0.07, dim=-1), dim=-1)  # (B, B, T)
        # col-wise softmax and element-wise multiply, then col-wise sum
        after_softmax_t2i = torch.nansum(dot_p * torch.softmax(dot_p/0.07, dim=-2), dim=-2)  # (B, B, L)

        # extend attention masks
        feature1_att_mask = feature1_att_mask.unsqueeze(1).repeat(1, features1.shape[0], 1).to(dot_p.device)  # (B, B, T)
        feature2_att_mask = feature2_att_mask.unsqueeze(0).repeat(features2.shape[0], 1, 1).to(dot_p.device)   # (B, B, L)
        after_softmax_i2t[~feature1_att_mask] = 0
        after_softmax_t2i[~feature2_att_mask] = 0

        logit_scale = passed_logit_scale.exp()

        # average and obtain similarity matrix
        I2T_sim = logit_scale * torch.nansum(after_softmax_i2t, dim=-1) / feature1_att_mask.sum(dim=-1)  # (B, B)
        T2I_sim = logit_scale * torch.nansum(after_softmax_t2i, dim=-1) / feature2_att_mask.sum(dim=-1)  # (B, B)

        return I2T_sim, T2I_sim
    
    def forward(self, src_input, tgt_input):
        input_image_features, image_features, final_image_mask = self.model_images(src_input)
        image_mask = (final_image_mask == 1)

        input_text_features = src_input['input_embed'].to(self.device)
        text_features = src_input['output_embed'].to(self.device)
        
        text_mask = (src_input['text_padding_mask'] == 1)
        
        I2T_sim_input, T2I_sim_input = self.cross_lingual_similarity(
            input_image_features, input_text_features, image_mask, text_mask, self.logit_scale_input
        )
        
        I2T_sim_output, T2I_sim_output = self.cross_lingual_similarity(
            image_features, text_features, image_mask, text_mask, self.logit_scale
        )
        
        assert I2T_sim_output.shape[0] == I2T_sim_input.shape[0], f'logits_per_image {I2T_sim_output.shape[0]} and logits_per_input_image {I2T_sim_input.shape[0]} should have the same batch size'

        logits = {
            'image_feat': input_image_features,
            'text_feat': input_text_features,
            'final_image_mask': final_image_mask
        }

        return I2T_sim_input, T2I_sim_input, I2T_sim_output, T2I_sim_output, logits

class FeatureExtracter(nn.Module):
    def __init__(self, frozen=False):
        super(FeatureExtracter, self).__init__()
        self.conv_2d = resnet() # InceptionI3d()
        self.conv_1d = TemporalConv(input_size=512, hidden_size=1024, conv_type=3)

        if frozen:
            print('Freezing the feature extractor')
            for param in self.conv_2d.parameters():
                param.requires_grad = False
            for param in self.conv_1d.parameters():
                param.requires_grad = False

    def forward(self,
                src: Tensor,
                src_length_batch
                ):
        src, skipconnect = self.conv_2d(src,src_length_batch)
        src = self.conv_1d(src)

        return src, skipconnect

class V_encoder(nn.Module):
    def __init__(self,
                 emb_size,
                 feature_size,
                 config,
                 ):
        super(V_encoder, self).__init__()
        
        self.config = config

        self.src_emb = nn.Linear(feature_size, emb_size)
        modules = []
        modules.append(nn.BatchNorm1d(emb_size))
        modules.append(nn.ReLU(inplace=True))
        self.bn_ac = nn.Sequential(*modules)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d,nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self,
                src: Tensor,
                ):
      
        src = self.src_emb(src)
        src = self.bn_ac(src.permute(0,2,1)).permute(0,2,1)

        return src

class SAGETranslator(nn.Module):
    def __init__(self, config, args, embed_dim=1024, pretrain=None, device=None):
        super(SAGETranslator, self).__init__()
        self.device = device
        self.config = config
        self.args = args

        self.backbone = FeatureExtracter(frozen=False)
    
        # self.mbart = config_decoder(config)
        self.mbart = MBartForConditionalGeneration.from_pretrained(config['model']['transformer'])
        self.vis_lang_proj = make_head(embed_dim, embed_dim, 'TV_Projector')
 
        self.sign_emb = nn.Identity()
        
    def share_forward(self, src_input):
        
        x, skipconnect = self.backbone(src_input['input_ids'].to(self.device), src_input['src_length_batch'])
        attention_mask = src_input['attention_mask']

        mask = attention_mask.to(self.device).unsqueeze(-1).float() 
        sum_embeddings = (x * mask).sum(dim=1)  # B x C
        sum_mask = mask.sum(dim=1)  # B x 1
        output_cls_rep = sum_embeddings / sum_mask  # B x C  

        output_cls_rep = self.vis_lang_proj(output_cls_rep)

        seg_ids = src_input['seg_ids']

        # Group embeddings by word ID
        grouped_embeddings = [torch.stack([output_cls_rep[i] for i in indices]) 
                            for _, indices in itertools.groupby(range(len(seg_ids)), key=lambda i: seg_ids[i])]

        padded_embeddings = pad_sequence(grouped_embeddings, padding_value=PAD_IDX, batch_first=True)
        final_attention_mask = (padded_embeddings[:, :, 0] != PAD_IDX).long()
        
        return padded_embeddings, final_attention_mask

    def forward(self,src_input, tgt_input):
        
        inputs_embeds, attention_mask = self.share_forward(src_input)

        out = self.mbart(inputs_embeds = inputs_embeds,
                    attention_mask = attention_mask.to(self.device),
                    # decoder_input_ids = tgt_input['input_ids'].cuda(),
                    labels = tgt_input['input_ids'].to(self.device),
                    decoder_attention_mask = tgt_input['attention_mask'].to(self.device),
                    return_dict = True,
                    )
        return out['logits']
    

    def generate(self,src_input,max_new_tokens,num_beams,decoder_start_token_id ):
        inputs_embeds, attention_mask = self.share_forward(src_input)

        out = self.mbart.generate(inputs_embeds = inputs_embeds,
                    attention_mask = attention_mask.to(self.device),max_new_tokens=max_new_tokens,num_beams = num_beams,
                                decoder_start_token_id=decoder_start_token_id
                            )
        return out