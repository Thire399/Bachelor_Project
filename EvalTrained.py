"""Imports"""

import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from torch import nn
from kornia.color import RgbToGrayscale
from PIL import Image
from PIL import ImageFilter
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import patchify as Pf
import os
from sklearn.metrics import precision_recall_curve
from torchvision import models as Vmodels
import Models
print('Done with imports')

"""# Data Loading functions

"""

def Get_Files(directory, num_files = 10):
    '''
    < Description > \n
        Gets the abs path of all files in a given directory
    < Description >\n
     -------------------------------------------------------------\n
    < Inputs >\n
     Directory to search\n
     Num_files: currently unused. Might be used to control number of files loaded in future version
    < Inputs >\n
     -------------------------------------------------------------\n
    < Return >\n
     Array of abs path for each file in the given directory.
    < Return >\n
    '''
    TempPath = []
    for dirpath, dirs, filenames in os.walk(directory):
        for f in filenames:
                TempPath.append(os.path.abspath(os.path.join(dirpath, f)))
    return TempPath

def create_pathces (Images, W, H, step_size):
    '''
    Takes in an Image of shape (4 x W X H)\n
    Does not handle 4 channels yet, so I trash the 4th. channel\n
    This should be superior compared to a costum implementation, as it should not use more memory when creating views.\n
    Just not as robost or costumizeable, if using own implementation\n
    Now handels overlap
    '''
    All_patches = []
    Images = np.asarray(Images)
    for i in range (Images.shape[0]):
        image = Images[i][:,:,:] #useless step?
        patches = Pf.patchify(image, (1, W, H), step= step_size)
        #Patches created = 11 x 11 x 512 x 512 with 3 channels. i.e. It is 11 rows and 11 columns. if step size = W and H, then no overlap
        All_patches.append(patches)

    return np.asarray(All_patches)

def Recreate_image(image, Ori_shape):
    '''
    Takes a bunch of patches, and puts them together to the original shape of the original image. REWRITE
    '''
    #print('Recreate images')
    #print('Input shape: ', image.shape)
    #print('output shape: ', Ori_shape)
    temp = []
    for i in range (image.shape[0]):
        temp.append(Pf.unpatchify(image[i], Ori_shape))
    return np.asarray(temp)

def swapaxes(X):
    '''
    Swaps axis to get correct input shape
    Also Gray scales an images as off now (28/04-2021)
    <Input>
     - X: A torch array of images. [N, W, H, C].
    ------------------------
    <Output>
     - A gray scaled image torch array of size [N, 1, W, H]
    '''
    X = torch.transpose(X, 1,-1)
    X = torch.transpose(X, 2,-1)
    gray = RgbToGrayscale()
    #plt.imshow(gray(X)[0][0],cmap = 'gray')
    #plt.show()
    return gray(X)


def Deconstruct(Patches):
    '''
    Must Take a list of images converted into patches.\n
    E.g.\n
    List shape: N x [n x Row x Col x C x W x H]
    '''
    ss = Patches[0].shape

    ndArr_X = []
    #Forced to use a train loader
    for N in range(len(Patches)): # N images
        for i in range (ss[1]): #How many Rows we have
                #print('The i\'th run:', i)
                for j in range (ss[2]): # How many Columns we have
                    img_new = np.asarray(Patches[N][0, i,j])
                    ndArr_X.append(img_new)
    ndArr_X = np.asarray(ndArr_X)

    #print('Output has type:', type(ndArr_X), ' With the shape: ', ndArr_X.shape)
    return ndArr_X

def Reconstruct(ndArr_X, Ori_PatchShape, N_img):
    '''
    Reconstructs the d-dimensional array given to the "Deconstruct" function\\
    This is then reconstructed in the original patch array shape
    '''
    #Creates corrects size matrix with

    Back_to_origianl = []
    for N in range (N_img):
        N_Image = np.zeros(Ori_PatchShape)
        #print('This is N', N)
        Nr_image = N * N_Image.shape[1] * N_Image.shape[2]
        for i in range (N_Image.shape[1]):
            for j in range (N_Image.shape[2]):
                index = (Nr_image+(i*N_Image.shape[1])+j)
                #print('Index: ', index)
                N_Image[0, i, j] = ndArr_X[index]
        Back_to_origianl.append(N_Image)
    #print('Reconstructed: ', len(Back_to_origianl), 'Image patch arrays')

    return np.asarray(Back_to_origianl)

def Evalf_test(model, loss, X, Y):
    '''
    For evaluating the model on Validation data
    '''
    out = model(X)
    loss_out = loss(out, Y)
    for i in range(out.shape[0]):
        temp = np.asarray(out[i].cpu().detach().numpy())
        temp = temp[0]
        im = Image.fromarray((temp*255).astype(np.uint8))
        im = im.save(f'Bachelor_Project/Data/out/Test/{i}_test.tif')
    return loss_out.item()

def To_Tuple(X, Y):
    '''
    Takes in X data and Y labels, and creates an array with tuble (x, y)
    for each element in the input data
    Not used right now.
    '''
    out = []
    for i in range(X.shape[0]):
        out.append((X[i], Y[i]))
    return np.asarray(out)

def From_Tuple(input):
    '''
    Takes in an Array of tuples, and output set X and set Y.
    Not used right now
    '''
    X = []
    Y = []
    for i in range(input.shape[0]):
        X.append(input[i][0])
        Y.append(input[i][1])
    return np.asarray(X), np.asarray(Y)

def PRAUC(pred, Target, ep):
    '''
    Takes in output from model and Target labels,
    and an epsilong to avoid div by 0
    Calculates the precision and recall for the given model over multiple
    thressholds, by using the sklearn implementation
    https://medium.com/@douglaspsteen/precision-recall-curves-d32e5b290248
    '''
    pred = pred.contiguous().view(-1)
    pred = pred.cpu().detach().numpy()
    Target = Target.contiguous().view(-1)
    Target = Target.cpu().detach().numpy()

    #rewrite into a function.
    P, R, T = precision_recall_curve(y_true = Target, probas_pred = pred)
    fscore = (2 * P * R) / (P + R + ep)

    # locate the index of the largest f score
    ix = np.argmax(fscore)
    bestT = T[ix]
    print('Best Threshold=%f, F-Score=%.3f' % (T[ix], fscore[ix]))

    no_skill = len(Target[Target == 1]) / len(Target)
    plt.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
    plt.plot(R, P, marker='.', label = 'Logistic')
    legend = plt.legend(loc = 'upper right', frameon = 1)
    frame = legend.get_frame()
    frame.set_facecolor('grey')
    frame.set_edgecolor('black')
    plt.xlabel ('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall curve')
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    plt.show()
    return bestT

def MinMaxNorm(x):
    '''
    "Unsqueezes" the input tensor values.
    Note
    Min-max normalization has one fairly significant downside: it does not handle outliers very well.
    '''
    max = torch.max(x)
    min = torch.min(x)
    temp = (x - min)/(max - min)
    return temp.detach().numpy()

"""# UNet model and diceloss"""

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

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding= 1)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding= 1)
        self.BN    = nn.BatchNorm2d(out_ch, affine=False)
        self.Drop  = nn.Dropout(0.10)

    def forward(self, x):
        return self.relu((self.BN (self.conv2( self.relu( self.BN(self.conv1(x)))))))

#This is the downsampling step/ the encoding step. We transform?
class Encoder(nn.Module):
    def __init__(self, chs=(4, 8, 16, 32, 64, 128)): # was 64,128,256,512,1024
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs

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

"""# Early stopping

"""

#The early stopping class is from here this github, and the credit goes to him.
# I only use it for early stopping.
#https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb
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


"""# SSL part """

#use the pretrained model, and augment data as above.
#implement the attention network F
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
        #for param in self.conv_block.parameters():
        #    param.requires_grad = False

        #self.conv_block.fc.requires_grad = True

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
        return_dict['fg_feats'] = h5
        return return_dict

class F_AttentionNet(nn.Module):
    '''
    Combines the pretrained Resnet 34, and an attention network.
    '''
    def __init__(self, num_class_Scale, flag = True, num_class = 3):
        super().__init__()
        self.AttentionNet = UNet((1, 4, 8, 16, 32),(32, 16, 8, 4), num_class)
        #self.AttentionNet = UNet((1, 2, 4, 8, 16),(16, 8, 4, 2), num_class)
        self.ResNet34 = PretrainedResNet34(flag, num_class_Scale)
        self.Conv1x1  = nn.Conv2d(3, num_class, 1, 1, 0) #only used if attention only
        self.num_class   = num_class
        '''
            # ----- *Simple MLP* ------

        self.relu           = nn.ReLU()
        self.linear1        = nn.Linear(1, 64, bias= False)
        self.linear2        = nn.Linear(64, 128, bias= False)
        self.linear3        = nn.Linear(128, 256, bias= False)
        self.avgpool        = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.last           = nn.Linear(256, num_S_Class, bias= True)
        '''
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
            '''
            out = torch.transpose(out, 1, -1)
            out = torch.transpose(out, 1, 2)
            return_dict = {}
            h1 = self.linear1(out)
            h1 = self.relu(h1)
            h2 = self.linear2(h1)
            h3 = self.relu(h2)
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
            '''
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
            Pred_Probs = self.ResNet34(F_attention)
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
            F_attention = self.softMax(out)
        return F_attention

'''
Default Values
------------------------------------------------------------------------------
'''
# Model definning
num_class = 1
num_S_Class = 2  #Number of scales
batch_size = 1
Epsilon = 1e-6
# Patch parameters
step_size = 1024
W, H = (1024, 1024)
#ModelName = 'MLP'
#ModelName = 'Model_1ResMonday'
ModelName = 'Model_TestLoss_0_57623_sgd'
#ModelName = 'No_Class_Imbalance'

'''
------------------------------------------------------------------------------
'''
DataDirPath = '/home/thire399/Documents/Bachelor_Project/Data/MonuSeg'
DL = diceloss()

DataPath = Get_Files(DataDirPath)

#channels = ((1, 64, 128, 256, 512, 1024), (1024, 512, 256, 128, 64))
#model = UNet(enc_chs = channels[0], dec_chs = channels[1], num_class = 1)
#model.load_state_dict(torch.load('/home/thire399/Documents/Bachelor_Project/Data/out/Baseline_Dataloader/Model_Eval_Shuffle'))

#model = ScaleNet(num_S_Class, flag = True, num_class = num_class)
#model.load_state_dict(torch.load('/home/thire399/Documents/Bachelor_Project/Data/out/SSL/SSLTuesdayBS10'))


#Validation Data
Validation_Data = torch.load(DataPath[2])
X_Val, Y_Val = Validation_Data
X_Val = swapaxes(X_Val)
Y_Val =  Y_Val.unsqueeze(1)
Patches_Val_X = create_pathces(X_Val, W, H, step_size = step_size)
X_Val_Data = Deconstruct(Patches_Val_X)
Patches_Val_Y = create_pathces(Y_Val, W, H, step_size = step_size)
Y_Val_Data = Deconstruct(Patches_Val_Y)

# Test Data
Test_Data = torch.load(DataPath[0])
X_test, Y_test = Test_Data
X_test = swapaxes(X_test)
Y_test =  Y_test.unsqueeze(1)

Patches_test_X = create_pathces(X_test, W, H, step_size = step_size)
X_Test_Data = Deconstruct(Patches_test_X)
Patches_test_Y = create_pathces(Y_test, W, H, step_size = step_size)
Y_Test_Data = Deconstruct(Patches_test_Y)

Val_Set = torch.utils.data.TensorDataset(torch.tensor(X_Val_Data),
                                                torch.tensor(Y_Val_Data))
Test_Set = torch.utils.data.TensorDataset(torch.tensor(X_Test_Data),
                                                torch.tensor(Y_Test_Data))

validation_loader = torch.utils.data.DataLoader(Val_Set,
                                        batch_size = batch_size,
                                        num_workers=0)

test_loader = torch.utils.data.DataLoader(Test_Set,
                                        batch_size = batch_size,
                                        num_workers=0)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using: ', device)
model = F_AttentionNet(num_S_Class, flag = True, num_class = num_class)
model.load_state_dict(torch.load(f'/home/thire399/Documents/Bachelor_Project/Data/out/SSLUNet/Model/{ModelName}'))

model.to(device)
Loss_per_imageVal = []
Loss_per_imageTest = []
Out_patchesVal =[]
with torch.no_grad():
    model.eval()
    for batch, (Data, Target) in enumerate(validation_loader, 1):
                Attention = model(Data.to(device), Only_Attention = True)
                #Saving each patch
                for i in range (Attention.shape[0]):
                    Out_patchesVal.append(Attention[i].cpu().detach().numpy())
images = Reconstruct(np.asarray(Out_patchesVal), Patches_Val_X[0].shape, 9)
images = Recreate_image(images, X_Val[0].shape)
images = MinMaxNorm(torch.from_numpy(images))
Threshold = PRAUC(torch.from_numpy(images), (Y_Val), Epsilon)
Threshold = 0.5
# ------------- ----------------------
images = images >= Threshold
images = images.astype(int)

for i in range (images.shape[0]):
    temp = np.asarray(images[i])
    temp = temp[0]
    im = Image.fromarray((temp*255).astype(np.uint8))
    im = im.filter(ImageFilter.ModeFilter(size=3))
    im = im.save(f'/home/thire399/Documents/Bachelor_Project/Data/out/SSLUNet/DSC_{i+1}.tif')

Loss_per_imageVal = []
for i in range(images.shape[0]):
    Loss_per_imageVal.append(1-DL(torch.from_numpy(images)[i],  Y_Val[i] ).item())
print(Loss_per_imageVal)


Out_patchesTest = []
Target_Labels = []
with torch.no_grad():
    model.eval()
    for batch, (Data, Target) in enumerate(test_loader, 1):
        Att = model(Data.to(device), Only_Attention = True)

        # for recreating the images
        for i in range (Att.shape[0]):
            Out_patchesTest.append(Att[i].cpu().detach().numpy())
            Target_Labels.append(Target[i].detach().numpy())

# for saving test the images
images = Reconstruct(np.asarray(Out_patchesTest),
                        Patches_test_X[0].shape, 14)
images = Recreate_image(images, X_test[0].shape)
images = MinMaxNorm(torch.from_numpy(images))
#Threshold = PRAUC(torch.from_numpy(images), (Y_test), Epsilon)

# ------------- ----------------------
images = images >= Threshold
images = images.astype(int)
for i in range(images.shape[0]):
    temp = np.asarray(images[i])
    temp = temp[0]
    im = Image.fromarray((temp*255).astype(np.uint8))
    im = im.filter(ImageFilter.ModeFilter(size=3))
    im = im.save(f'/home/thire399/Documents/Bachelor_Project/Data/out/SSLUNet/Test/{i+1}_Test.tif')

for i in range(images.shape[0]):
    Loss_per_imageTest.append(1-DL(torch.from_numpy(images)[i],  Y_test[i] ).item())
print(Loss_per_imageTest)
