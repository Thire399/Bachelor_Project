"""Imports"""
from numpy.core.fromnumeric import sort
import torch
from torch import nn
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision import models as Vmodels
import cv2
import re
import os
import numpy as np

def sorted_alphanumeric(data):
    '''
    https://stackoverflow.com/questions/4813061/non-alphanumeric-list-order-from-os-listdir
    '''
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

def Load_all_files_dir(path):
    '''
    Loads all files in a Google drive directory.
    '''
    temp_arr = []
    for image in sorted_alphanumeric(os.listdir(path)):
        img_path = path + '/'+ image
        temp_arr.append(img_path)
    return temp_arr

class diceloss(torch.nn.Module):
    def init(self):
        super(diceLoss, self).init()
    def forward(self,pred, target):
        smooth = 1.
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        return 1 -((2. * intersection + smooth) / (A_sum + B_sum + smooth))

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class DataSet(torch.utils.data.Dataset):
    def __init__(self, data_dir = './', transform = None):
        super().__init__()
        self.data_dir = data_dir
        self.Target = torch.load(f'{data_dir}/Labels/PseudoLabels.pt')
        self.transform = transform
    def __len__(self):
        return len(self.Target)

    def __getitem__(self, index):
        label = self.Target[index]
        img_path = Load_all_files_dir(f'{self.data_dir}/Images')
        image = cv2.imread(img_path[index], cv2.IMREAD_GRAYSCALE)
        # cv2.CV_LOAD_IMAGE_GRAYSCALE flag is needed to let it know it is grayscaled

        if self.transform is not None:
            image = self.transform(image)
        return image, label


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding= 1)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding= 1)
        self.BN    = nn.BatchNorm2d(out_ch, affine=False)
        self.Drop  = nn.Dropout(0.20)

    def forward(self, x):
        return self.relu(self.conv2( self.relu( self.conv1(x))))

#This is the downsampling step/ the encoding step. We transform?
class Encoder(nn.Module):
    def __init__(self, chs=(4, 8, 16, 32, 64, 128)): # was 64,128,256,512,1024
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)

    def forward(self, x):
        Features = []
        for block in self.enc_blocks:
            x = block(x)
            Features.append(x)
            x = self.pool(x)
        return Features

#This is the decoder, or where we upsample again, / putting everything together
class Decoder(nn.Module):
    def __init__(self, chs=(128, 64, 32, 16, 8)): # was 1024, 512, 256, 128, 64
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2, padding=0) for i in range(len(chs)-1)]) #maybe use torch unpool "max unpool 2D"
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs

#The final UNet implementation
#was def __init__(self, enc_chs=(3,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1, retain_dim=False, out_sz=(572,572)):
class UNet(nn.Module):
    def __init__(self, enc_chs=(3, 8, 16, 32, 64, 128), dec_chs=(128, 64, 32, 16, 8), num_class=1, retain_dim=True): #Change num_class to handle 4 channels
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)#,padding=5) #consider why using padding of 5, best with no padding here. would be better in the decoding step
        self.retain_dim  = retain_dim
        self.sig         = nn.Sigmoid() #clamps the output to between 1 and 0
        self.softMax     = nn.Softmax2d() #clamps output between 1 and 0. differently from the sigmoid
        self.num_class   = num_class #think of it as the number of objects to segment


    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        if (self.num_class == 1):
            out      = self.sig(out)
        else:
            out = self.softMax(out)
        if self.retain_dim:
            out = F.interpolate(out, (x.shape[2],x.shape[3]), mode = 'nearest') #nearest #Bicubic should have less artifacts
        return out


class DilatedConvBlock(nn.Module):
    def __init__(self, in_c = 1, out_c = 32, **kwargs):
        super(DilatedConvBlock, self).__init__()
        #using Sequential to keep forward function small
        self.main = nn.Sequential(
            nn.Conv2d(in_c, out_c, **kwargs),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True),
        )
    def forward(self, x):
        return self.main(x)

class DilatedConvResBlocks(nn.Module):
    '''
    Uses the same structure as a resnet conv layer block.
    '''
    def __init__(self, in_c = 1, out_c = 32, **kwargs):
        super(DilatedConvResBlocks, self).__init__()
        self.res    = nn.Sequential(
            nn.Conv2d(in_c, out_c, **kwargs),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True),

            nn.Conv2d(out_c, out_c, **kwargs),
            nn.BatchNorm2d(out_c),)

        if out_c != in_c:
        # Mapping connection.
            self.mapper = nn.Sequential(
                        nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0, bias=False),
                        nn.BatchNorm2d(out_c), )
        self.relu   = nn.ReLU(True)
        self.in_c = in_c
        self.out_c = out_c

    def forward(self, x, **kwargs):
        residual = self.res(x)
        if self.in_c != self.out_c:
            x           = self.mapper(x)
        out             = self.relu(x + residual)
        return out

'''
This is the Attention network as proposed by https://arxiv.org/pdf/2007.08373.pdf
It consist of 15 Convolutional layers.
'''

class PretrainedResNet34(nn.Module):
    def __init__(self, flag, num_class):
        super(PretrainedResNet34, self).__init__()
        self.conv_block         = Vmodels.resnet34(pretrained = flag)
        self.conv_block.fc      = nn.Linear(512, num_class, bias = True) #making it output the number of classes.
        self.conv_block.conv1   = nn.Conv2d (1, 64, (7, 7), (2, 2),
                                            padding = (3, 3), bias = False )

    def forward(self, x, **kwargs):
        return_dict = {}
        x = self.conv_block.conv1(x)
        x = self.conv_block.bn1(x)
        x = self.conv_block.relu(x)
        h0 = self.conv_block.maxpool(x)

        h1 = self.conv_block.layer1(h0)
        h2 = self.conv_block.layer2(h1)
        h3 = self.conv_block.layer3(h2)
        h4 = self.conv_block.layer4(h3)
        h4_1 = self.conv_block.avgpool(h4)
        h4_1 = torch.flatten(h4_1, 1)
        h5 = self.conv_block.fc(h4_1)

        if hasattr(kwargs, 'return_h_feats') and kwargs['return_h_feats']:
            return_dict['feats_0']  = h0
            return_dict['feats_1']  = h1
            return_dict['feats_2']  = h2
            return_dict['feats_3']  = h3
            return_dict['feats_4']  = h4
        return_dict['fg_feats'] = (h5)
        return return_dict


class ScaleUNet_MLP(nn.Module):
    '''
    Combines the pretrained Resnet 34, and an attention network.
    '''
    def __init__(self, num_S_Class, flag, num_class = 3):
        super().__init__()
        self.AttentionNet   = UNet((1, 4, 8, 16, 32),(32, 16, 8, 4), num_class)
        self.Conv1x1        = nn.Conv2d(2, num_class, 1, 1, 0) #only used if attention only
        self.num_class      = num_class

        # ----- *Simple MLP* ------

        self.relu           = nn.ReLU()
        self.linear1        = nn.Linear(1, 64, bias= False)
        self.linear2        = nn.Linear(64, 128, bias= False)
        self.linear3        = nn.Linear(128, 256, bias= False)
        self.avgpool        = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.last           = nn.Linear(256, num_S_Class, bias= True)

    def forward(self, x, Only_Attention = False):
        '''
        If Only_Attention == true
            Returns only the attentionmap
        Else
            Returns both the Predicted class probabilities,
            and the attentionmap.
        '''
        out = self.AttentionNet(x)
        if Only_Attention == False:
            #Pred_Probs = self.ResNet34(out)

            out = torch.transpose(out, 1, -1)
            out = torch.transpose(out, 1, 2)
            return_dict = {}
            h1 = self.linear1(out)
            h1 = self.relu(h1)
            h2 = self.linear2(h1)
            h2 = self.relu(h2)
            h3 = self.linear3(h2)
            h3 = self.relu(h3)
            h4_1 = self.avgpool(h3)
            h4_1 = torch.flatten(h4_1, 1)
            h5 = self.last(h4_1)

            return_dict['feats_1']  = h1
            return_dict['feats_2']  = h2
            return_dict['feats_3']  = h3
            return_dict['fg_feats'] = h5

            Pred_Probs = return_dict
            out = torch.transpose(out, 1, -1)
            out = torch.transpose(out, -1, 2)

            return out, Pred_Probs

        return out


class PretrainedResNet34_Freeze(nn.Module):
    def __init__(self, flag, num_class):
        super(PretrainedResNet34_Freeze, self).__init__()
        self.conv_block         = Vmodels.resnet34(pretrained = flag)
        self.conv_block.fc      = nn.Linear(512, num_class, bias = True) #making it output the number of classes.
        self.conv_block.conv1   = nn.Conv2d (1, 64, (7, 7), (2, 2),
                                            padding = (3, 3), bias = False )
        for param in self.conv_block.parameters():
            param.requires_grad = False

        self.conv_block.fc.requires_grad = True

    def forward(self, x, **kwargs):
        return_dict = {}
        x = self.conv_block.conv1(x)
        x = self.conv_block.bn1(x)
        x = self.conv_block.relu(x)
        h0 = self.conv_block.maxpool(x)

        h1 = self.conv_block.layer1(h0)
        h2 = self.conv_block.layer2(h1)
        h3 = self.conv_block.layer3(h2)
        h4 = self.conv_block.layer4(h3)
        h4_1 = self.conv_block.avgpool(h4)
        h4_1 = torch.flatten(h4_1, 1)
        h5 = self.conv_block.fc(h4_1)

        if hasattr(kwargs, 'return_h_feats') and kwargs['return_h_feats']:
            return_dict['feats_0']  = h0
            return_dict['feats_1']  = h1
            return_dict['feats_2']  = h2
            return_dict['feats_3']  = h3
            return_dict['feats_4']  = h4
        return_dict['fg_feats'] = (h5)
        return return_dict

class ScaleUNet_Freeze(nn.Module):
    '''
    Combines the pretrained Resnet 34, and an attention network.
    '''
    def __init__(self, num_class_Scale, flag = True, num_class = 3):
        super().__init__()
        self.AttentionNet   = UNet((1, 4, 8, 16, 32),(32, 16, 8, 4), num_class)
        self.ResNet34 = PretrainedResNet34_Freeze(flag, num_class_Scale)
        self.num_class      = num_class

    def forward(self, x, Only_Attention = False):
        '''
        If Only_Attention == true
            Returns only the attentionmap
        Else
            Returns both the Predicted class probabilities,
            and the attentionmap.
        '''
        out = self.AttentionNet(x)
        if Only_Attention == False:
            Pred_Probs = self.ResNet34(out)

            return out, Pred_Probs

        return out

class ScaleUNet(nn.Module):
    '''
    Combines the pretrained Resnet 34, and an attention network.
    '''
    def __init__(self, num_class_Scale, flag = True, num_class = 3):
        super().__init__()
        self.AttentionNet   = UNet((1, 4, 8, 16, 32),(32, 16, 8, 4), num_class)
        self.ResNet34 = PretrainedResNet34(flag, num_class_Scale)
        self.num_class      = num_class

    def forward(self, x, Only_Attention = False):
        '''
        If Only_Attention == true
            Returns only the attentionmap
        Else
            Returns both the Predicted class probabilities,
            and the attentionmap.
        '''
        out = self.AttentionNet(x)
        if Only_Attention == False:
            Pred_Probs = self.ResNet34(out)

            return out, Pred_Probs

        return out

class ScaleNet(nn.Module):
    '''
    Combines the pretrained Resnet 34, and an attention network.
    '''
    def __init__(self, num_class_Scale, in_c = 1, out_c = 64, flag = True, num_class = 3):
        super().__init__()
        self.AttentionNet = nn.Sequential(
                DilatedConvBlock(in_c, out_c, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                DilatedConvResBlocks(out_c, out_c, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                DilatedConvResBlocks(out_c, out_c, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
                DilatedConvResBlocks(out_c, out_c, kernel_size=3, stride=1, padding=3, dilation=3, bias=False),
                DilatedConvResBlocks(out_c, out_c, kernel_size=3, stride=1, padding=5, dilation=5, bias=False),
                DilatedConvResBlocks(out_c, out_c, kernel_size=3, stride=1, padding=10, dilation=10, bias=False),
                DilatedConvResBlocks(out_c, out_c, kernel_size=3, stride=1, padding=20, dilation=20, bias=False),
                nn.Conv2d(out_c, 3, kernel_size=1, stride=1, padding=0), )

        self.ResNet34 = PretrainedResNet34(flag, num_class_Scale)
        self.Conv1x1  = nn.Conv2d(3, num_class, 1, 1, 0) #only used if attention only
        self.relu  = nn.ReLU()
        self.sig         = nn.Sigmoid() #clamps the output to between 1 and 0
        self.softMax     = nn.Softmax2d() #clamps output between 1 and 0. differently from the sigmoid
        self.num_class   = num_class

    def forward(self, x, Only_Attention = False):
        '''
        If Only_Attention == true
            Returns only the attentionmap
        Else
            Returns both the Predicted class probabilities,
            and the attentionmap.
        '''
        F_attention = self.AttentionNet(x)

        if Only_Attention == False:
            Pred_Probs = self.ResNet34(x * F_attention)
            F_attention = self.Conv1x1(F_attention) #Should only change the filter dimensionality.

            if (self.num_class == 1):
               F_attention      = self.sig(F_attention)
            else:
              F_attention = self.softMax(F_attention)

            return F_attention, Pred_Probs

        F_attention = self.Conv1x1(F_attention) #Should only change the filter dimensionality.
        if (self.num_class == 1):
            F_attention      = self.sig(F_attention)
        else:
            F_attention = self.softMax(F_attention)
        return F_attention
