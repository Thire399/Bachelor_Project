"""Imports"""
import os
import numpy as np
import torch
from torch import nn
from kornia.color import RgbToGrayscale
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import patchify as Pf
import os
from collections import Counter
print('Done with imports')
"""# Data Loading functions

"""
### Defining Data augmentation functions
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

def Transform(X, Y, If_trainX = False):
    '''
    Takes in arrays of images.
    Then uses torch functions to
       - Flip (mirror the image) in the Y axis
       - Flip (mirror the image) in the x axis
       - Rotate by 90 degrees
    -------------------------------
    This is done for both X and Y, so they correspond.
    Thus it returns both X and Y.
    '''
    x_outArr = []
    y_outArr = []

    for i in range(X.shape[0]):
        #Transform for X
        x_outArr.append(X[i][0].unsqueeze(0).unsqueeze(0))
        x_outArr.append(X[i][0].flip(1).unsqueeze(0).unsqueeze(0)) #Filps in y axis
        x_outArr.append(X[i][0].flip(0).unsqueeze(0).unsqueeze(0)) #Filps in x axis
        x_outArr.append(X[i][0].rot90(k = 1, dims = (0, 1)).unsqueeze(0).unsqueeze(0))

        if If_trainX == False:
            #Transform for y
            y_outArr.append(Y[i].unsqueeze(0))
            y_outArr.append(Y[i].flip(1).unsqueeze(0))
            y_outArr.append(Y[i].flip(0).unsqueeze(0))
            y_outArr.append(Y[i].rot90(k = 1, dims = (0, 1)).unsqueeze(0))
    x_outArr = torch.cat(x_outArr, dim = 0)
    if If_trainX == False:
        y_outArr = torch.cat(y_outArr, dim = 0)
        return x_outArr, y_outArr
    else:
        return x_outArr

def UpScale(img, scale):
    '''
    Img: Only performs scaling to H and W, by HxS, WxS where S is scale.
    Scale: the scale number. Also the label for resnet later.

    '''
    scaler = nn.Upsample(scale_factor = (scale, scale), mode ='nearest')
    out = scaler(img)
    return out

def generateNewData(Images, n_scale, W, H, step_size, SavePath, IfVal = False):
    '''
    generateNewData handles scaling the data and Data augmentation.
    Input:
        - Images :
        - n_scale : number of classes
        - Patch_size : Tuple of (H, W)
        - If_Val : Bool Validation/Test set
    ----------------
    Returns:
        - Scaled X data
        - Labels for sacled data
    '''
    #Use equal number of datapoint for each scale to avoid class imbalance
    scale = 1
    y_scale_out = []
    patch_num = 0 #It will not read correctly unless using this
    if (IfVal == True):
        print('    ---- Validation ----')
        for i in range(n_scale):
            tempSave = []
            if (scale == 1):
                TempX = create_pathces(Images.detach().numpy(), W, H, step_size)
                TempXShape = TempX.shape
                size = TempXShape[0]*TempXShape[2]*TempXShape[3] #number of patches
                X_out = Deconstruct(TempX)
            else:
                temp = UpScale(Images, scale = scale)
                TempX = create_pathces(temp.detach().numpy(), W, H, step_size)
                X_out = Deconstruct(TempX)
                idx = np.random.randint(low =0, high = X_out.shape[0], size = size)
                for j in range(idx.shape[0]):
                    tempSave.append(np.asarray([X_out[j]]))
                X_out = tempSave
                X_out = np.vstack(X_out)

            for j in range (X_out.shape[0]):

                temp = np.asarray(X_out[j])
                temp = temp[0]
                im = Image.fromarray((temp*255).astype(np.uint8))
                im = im.save(f'{SavePath}/Images/Img_Scale_{scale}_Patch_{j}.tif')
                patch_num += 1

            y = torch.from_numpy(np.asarray([scale-1]*size))
            y_scale_out.append(y)
            scale +=1
            print(f'Done for Scale class: {i+1}')

    else:
        print('    ---- Train ----')
        for i in range(n_scale):
            tempSave = []
            if (scale == 1):
                temp = Transform(Images, None, If_trainX = True)
                TempX = create_pathces(temp.detach().numpy(), W, H, step_size)
                TempXShape = TempX.shape
                size = TempXShape[0]*TempXShape[2]*TempXShape[3] #number of patches
                X_out = Deconstruct(TempX)

                #double the first class to level out some class imbalance
            elif(scale == 2):
                #Subsample. (Select the number of samples needed)
                temp = UpScale(Images, scale = scale)
                TempX = create_pathces(temp.detach().numpy(), W, H, step_size)
                X_out = Deconstruct(TempX)
            else:
                temp = UpScale(Images, scale = scale)
                TempX = create_pathces(temp.detach().numpy(), W, H, step_size)
                X_out = Deconstruct(TempX)
                idx = np.random.randint(low =0, high = X_out.shape[0], size = size)
                for j in range(idx.shape[0]):
                    tempSave.append(np.asarray([X_out[j]]))
                X_out = tempSave
                X_out = np.vstack(X_out)

            for j in range (X_out.shape[0]):
                temp = np.asarray(X_out[j])
                temp = temp[0]
                im = Image.fromarray((temp*255).astype(np.uint8))
                im = im.save(f'{SavePath}/Images/Img_Scale{scale}_Patch{j}.tif')

            y = torch.from_numpy(np.asarray([scale-1]*size))
            y_scale_out.append(y)
            scale +=1
            print(f'Done for Scale class: {i+1}')

    y_scale_out = torch.cat(y_scale_out, 0)
    torch.save(y_scale_out, f'{SavePath}/Labels/PseudoLabels.pt')

    return y_scale_out


def ScalePatch(X_data, Y_labels, W, H, step_size):
    '''
    Takes in images and creates patches of the images, and correpsonding scale labels.\n
    Input:
        - X_data : images
        - Y_labels : a list of the label values.
        - W : Width of patch
        - H : Height of patch
        - step_size : how much to move the window for each patch.\n
    Returns:
        - X_out: Patches of the input data
        - y_scale_True : A list of correct labels
        - n_patches_shape: Saves the shape of each patch
    '''
    TempX = []
    X_out = []
    #Create Patches of the input images.
    for i in range(len(X_data)):
        TempX.append(create_pathces(X_data[i], W, H, step_size))
        X_out.append(torch.from_numpy(Deconstruct(TempX[i])))
    X_out = torch.cat(X_out, 0)
    #Create labels for each patch.
    y_scale_True = []
    n_patches_shape = [] #an array for the number of patches
    for i in Y_labels:
        '''
        Patches have shape [n, 1, Hp, Wp, H, W]
        Hp: is the number of patches along y-xis
        Wp: is the number of patches along x-axis
        '''
        tmp = TempX[i-1]
        n_patches_shape.append(tmp.shape)
        n = (tmp.shape[2]**2)*tmp.shape[0] #Assumes that H = W
        #Labels are i-1, as probabilities array are 0 index.
        y_scale_True.append(np.asarray([i-1]*n))

    y_scale_True = torch.from_numpy(np.concatenate(y_scale_True, axis = 0)).long()

    return X_out, y_scale_True, n_patches_shape

"""### Calling functions"""

'''
Default Values
------------------------------------------------------------------------------
'''
# Model definning
num_class = 1
num_S_Class = 5 #Number of scales

# Patch parameters
step_size = 256
W, H = (256, 256)
'''
------------------------------------------------------------------------------
'''

# ----------------------------------------------
#Train Data
#Loading data in.
DataDirPath = 'Some_Save_Path/MonuSeg'#Add path
DataPath = Get_Files(DataDirPath)

# ----------------------------------------------
#Train Data
Train_Data = torch.load(DataPath[1])
X_train, Y_train = Train_Data
X_train = swapaxes(X_train)
#Validation Data
Validation_Data = torch.load(DataPath[2])
X_Val, Y_Val = Validation_Data
X_Val = swapaxes(X_Val)

labels = generateNewData(X_train,
                                num_S_Class,
                                W, H,
                                step_size = step_size,
                                SavePath = 'Some_Save_Path/CostumDataSetTrain',#Add path
                                IfVal = False)
print(labels.shape, labels.unique())

c = Counter(labels.detach().numpy())
print(f'Counter for Train\n{c}')

labels = generateNewData(X_Val,
                                num_S_Class,
                                W, H,
                                step_size = step_size,
                                SavePath = 'Some_Save_Path/CostumDataSetVal',#Add path
                                IfVal = True)
print(labels.shape, labels.unique())

c = Counter(labels.detach().numpy())
print(f'Counter for Val\n{c}')