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
#When not training, it handels full size images.
step_size = 1024
W, H = (1024, 1024)
Chosen_model = Models.ScaleUNet
ModelName = 'ScaleUNet_MLP' #Example

'''
------------------------------------------------------------------------------
'''
DataDirPath = 'Path' #Add path to MoNuSeg data set.
DL = Models.diceloss()

DataPath = Get_Files(DataDirPath)

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

#Loading in the saved model weights
model = Chosen_Model(num_S_Class, flag = True, num_class = num_class)
model.load_state_dict(torch.load(f'Path_To_model/{ModelName}')) # Add Path
model.to(device)

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
    im = im.save(f'Some_Save_Path/DSC_{i+1}.tif') # Add Path


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
Threshold = PRAUC(torch.from_numpy(images), (Y_test), Epsilon)

# ------------- ----------------------
images = images >= Threshold
images = images.astype(int)
for i in range(images.shape[0]):
    temp = np.asarray(images[i])
    temp = temp[0]
    im = Image.fromarray((temp*255).astype(np.uint8))
    im = im.filter(ImageFilter.ModeFilter(size=3))
    im = im.save(f'Some_Save_Path/Test/{i+1}_Test.tif') # Add path
