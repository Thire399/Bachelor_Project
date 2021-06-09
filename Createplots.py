import PIL
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import re
from PIL import Image
import patchify as Pf
#x = Image.open('/home/thire399/Documents/Bachelor_Project/testImages/1_Test_BaseLine.tif')
#print(np.asarray(x).shape)

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

'''
DataDirPath = '/home/thire399/Documents/Bachelor_Project/Data/MonuSeg'

DataPath = Get_Files(DataDirPath)
print(DataPath)
Data = torch.load(DataPath[1])

#Data = torch.transpose(Data[0], 1,-1)
#Data = torch.transpose(Data, 2,-1)
Data = Data[0].detach().numpy()
#images = [0, 7, 10]
images = [0]
for i in images:
    print(i, Data[i].shape)
    im = Image.fromarray((Data[i]*255).astype(np.uint8))
    #im = im.convert('RGB')
    #im = im.save(f'/home/thire399/Documents/Bachelor_Project/testImages/{i+1}_Test_AOriginal.tif')
    #im = im.save(f'/home/thire399/Documents/Bachelor_Project/testImages/{i+1}_Test_AOriginal.tif')
    im = im.save(f'/home/thire399/Documents/Bachelor_Project/{i}_trainImage.tif')
'''
def saveimage(LossOrNot):
    if LossOrNot == False:
        Name = 'Useless'
        path = Load_all_files_dir(f'/home/thire399/Documents/Bachelor_Project/{Name}')
        fig = plt.figure(figsize=(12, 12))
        fig.tight_layout()

        #important to change! Controls layout of the figures

        # ax enables access to manipulate each of subplots
        temp = []
        ax = []
        for i in path:
            im = Image.open(i)
            im = np.asarray(im)

            if len(im.shape) == 2:
                im = np.expand_dims(im,-1)
            temp.append(im)
        '''
        columns = 5
        rows = 3
        notBW = [0, 5, 10]
        name = ['Original test\n image 1', 'Baseline test\n image 1', 'ScaleNet test\n image 1', 'ScaleUNet test\n image 1', 'Ground truth test\n image 1',
                'Original test\n image 8', 'Baseline test\n image 8', 'ScaleNet test\n image 8', 'ScaleUNet test\n image 8', 'Ground truth test\n image 8',
                'Original test\n image 11', 'Baseline test\n image 11', 'ScaleNet test\n image 11', 'ScaleUNet test\n image 11', 'Ground truth test\n image 11'] #update names for SSLNet
        '''
        columns = 4
        rows = 3
        notBW = [0, 4, 8]
        name = ['Original test image 1', 'MLP test image 1', 'Frozen test image 1', 'Ground truth test image 1',
                'Original test image 8', 'MLP test image 8', 'Frozen test image 8', 'Ground truth test image 8',
                'Original test image 11', 'MLP test image 11', 'Frozen test image 11', 'Ground truth test image 11'] #update names for SSLNet

        for i in range(len(temp)):
            ax.append( fig.add_subplot(rows, columns, i+1) )
            ax[i].set_title(name[i])
            ax[i].xaxis.set_visible(False)
            ax[i].yaxis.set_visible(False) # set title
            if i in notBW:
                plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
                plt.imshow(temp[i])
            else:
                plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
                plt.imshow(temp[i], cmap = 'gray' )
        plt.savefig(f'/home/thire399/Documents/Bachelor_Project/{Name}', dpi = 400, bbox_inches='tight')

        plt.show()
    else:
        path = Load_all_files_dir(f'/home/thire399/Documents/Bachelor_Project/LossPrEpoch')
        fig = plt.figure(figsize=(15, 15))

        #important to change! Controls layout of the figures
        columns = 3
        rows = 1

        # ax enables access to manipulate each of subplots
        temp = []
        ax = []
        for i in path:
            im = Image.open(i)
            im = np.asarray(im)

            if len(im.shape) == 2:
                im = np.expand_dims(im,-1)
            temp.append(im)

        name = ['Baseline UNet', 'ScaleNet', 'ScaleUNet'] #update names for SSLNet
        for i in range(len(temp)):
            ax.append( fig.add_subplot(rows, columns, i+1) )
            ax[i].set_title(name[i])  # set title
            plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
            plt.axis('off')
            plt.imshow(temp[i])

        plt.savefig('/home/thire399/Documents/Bachelor_Project/LossPrEpoch', dpi = 400, bbox_inches='tight')
        plt.show()

def SaveTrain():

    Name = 'PatchPlot'
    path = Load_all_files_dir(f'/home/thire399/Documents/Bachelor_Project/{Name}')
    fig = plt.figure(figsize=(12, 12))
    fig.tight_layout()

    #important to change! Controls layout of the figures

    # ax enables access to manipulate each of subplots
    temp = []
    ax = []
    for i in path:
        im = Image.open(i)
        im = np.asarray(im)

        if len(im.shape) == 2:
            im = np.expand_dims(im,-1)
        temp.append(im)

    columns = 3
    rows = 2
    notBW = [0, 4, 8]
    name = ['Original train image 1', 'Scale 1', 'Scale 2', 'Scale 3', ' scale 4', 'Scale 5'] #update names for SSLNet

    for i in range(len(temp)):
        ax.append( fig.add_subplot(rows, columns, i+1) )
        ax[i].set_title(name[i])
        ax[i].xaxis.set_visible(False)
        ax[i].yaxis.set_visible(False) # set title
        plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
        plt.imshow(temp[i], cmap = 'gray')
    plt.savefig(f'/home/thire399/Documents/Bachelor_Project/Scales.png', bbox_inches='tight')

    plt.show()

#saveimage(LossOrNot = True)
#SaveTrain()


im = Image.open(f'/home/thire399/Documents/Bachelor_Project/Data/out/SSLUNet/Test/8_Test.tif')
im = np.asarray(im)
im = np.expand_dims(im,0)
im = np.expand_dims(im,0)

temp = im
print(temp.shape)
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

x = create_pathces(temp, 256, 256, 256)
x = Deconstruct(x)
for i in range(len(x)):
    temp = np.asarray(x[i])
    im = Image.fromarray((temp*255).astype(np.uint8))
    im = im.save(f'/home/thire399/Documents/Bachelor_Project/Data/out/{i+1}.tif')
