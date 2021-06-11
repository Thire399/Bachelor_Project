'''
Code to Run the training scrip.

Remember to change path for the data set and where the images and the model is saved.
'''

"""Imports"""
import matplotlib.pyplot as plt
import os
import numpy as np
from numpy.core.fromnumeric import sort
import torch
import torch.optim as optim
from torch import nn
from kornia.color import RgbToGrayscale
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import patchify as Pf
import os
from sklearn.metrics import precision_recall_curve
from torchvision import models as Vmodels
import Models
print('Done with imports')

'''
Default Values
------------------------------------------------------------------------------
'''
ChoseModel = Models.ScaleUNet
# Model definning
num_class = 1
num_S_Class = 5 #Number of scales

# Training loop parameters
epoch = 100
Loss_Fun = nn.CrossEntropyLoss()
Optimizer = optim.Adam
learning_rate = 1e-4
batch_size = 12

# For early stopping
Patience = 10
Delta = 0.01
Epsilon = 1e-3

# Patch parameters
step_size = 256
W, H = (256, 256)

'''
------------------------------------------------------------------------------
'''

"""# Data Loading functions"""

def Get_Files(directory):
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
    Now handels overlap.
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
    plt.pause(3)
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

#Figure out smart way for this in a function
#Or simply call inside trainloop function before loop
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

def transform(images):
    image = torch.from_numpy(images)
    image = image.unsqueeze(0).int()
    image = torch.div(image, 255.0) #normalize

    return image

def Train_loop_SSL(model, Train_loader, Val_loader,
                   Test_loader, epochs, Loss_Fun, Op, Lr,
                   Out_Patch_shape, No_patch_shape, Val_Patches, Ori_Val_shape,
                   Epsilon, delta, patience, name):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using: ', device)
    if device == 'cpu':
        print('Not gonna run on the CPU')
        return None, None

    model.to(device)
    loss_function = Loss_Fun
    optimizer = Op(model.parameters(), lr = Lr)

    Train_loss = []
    Val_loss = []
    DL = Models.diceloss()
    print(f'Choosen model: {str(name)}')
    Model_Name = f'Some_Save_Path/{str(name)[15:-2]}' #Add path to save model.

    early_stopping = Models.EarlyStopping( patience=patience,
                        verbose=True, delta = delta, path =
                        Model_Name)

    for Epoch in range(epochs):
        Out_patchesVal = []
        BatchTrain_loss = []
        BatchTest_loss = []
        BatchVal_loss = []

        model.train(mode=True)
        for batch, (Data, Target) in enumerate(Train_loader, 1):
            optimizer.zero_grad() # a clean up step for PyTorch
            _, Pred = model(Data.to(device), Only_Attention = False)
            loss = loss_function(Pred['fg_feats'], (Target).to(device))
            loss.backward()
            optimizer.step()

            BatchTrain_loss.append(loss.item())  # each epoch loss
            if batch % 50 == 0:
                print('Train Epoch [{}/{}]: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    Epoch+1, epochs, batch * len(Data), len(Train_loader.dataset),
                    100. * batch / len(Train_loader),
                    np.mean(BatchTrain_loss)))
        Train_loss.append(np.mean(BatchTrain_loss))

        #-------------------- Validation data -----------------------
        model.eval()
        for batch, (Data, Target) in enumerate(Val_loader, 1):
            Attention, Pred = model(Data.to(device), Only_Attention = False)
            loss = loss_function(Pred['fg_feats'], (Target).to(device))
            BatchVal_loss.append(loss.item())
            if batch % 30 == 0:
                print(4*' ','===> Validation: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    batch * len(Data), len(Val_loader.dataset),
                    100. * batch / len(Val_loader),
                    np.mean(BatchVal_loss)))


        temp = np.mean(BatchVal_loss)
        Val_loss.append(temp)

        early_stopping(temp, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
        else:
            continue

    # ------------ Save Validation map using best model ---------------
    model.load_state_dict(torch.load(Model_Name))
    model.eval()
    for batch, (Data, Target) in enumerate(Val_loader, 1):
        Attention, Pred = model(Data.to(device), Only_Attention = False)
        loss = loss_function(Pred['fg_feats'], (Target).to(device))
        BatchVal_loss.append(loss.item())

        #Saving each patch
        for i in range (Attention.shape[0]):
            Out_patchesVal.append(Attention[i].cpu().detach().numpy())
    # ------------------- Recreate and do PRAUC -----------------------

    images = Reconstruct(np.asarray(Out_patchesVal), Val_Patches[0].shape, 9)
    images = Recreate_image(images, Ori_Val_shape)
    images = MinMaxNorm(torch.from_numpy(images))
    Threshold = PRAUC(torch.from_numpy(images), torch.from_numpy(Val_Patches), Epsilon)

    images = images >= Threshold
    images = images.astype(int)
    for i in range (images.shape[0]):
        temp = np.asarray(images[i])
        temp = temp[0]
        im = Image.fromarray((temp*255).astype(np.uint8))
        im = im.save(f'Some_Save_Path/DSC_{i+1}.tif') # Add path

    #-------------------- Test data -----------------------
    Out_patchesTest = []
    Target_Labels = []
    for batch, (Data, Target) in enumerate(Test_loader, 1):
        Att = model(Data.to(device), Only_Attention = True)
        loss = DL(Att, Target.to(device))

        BatchTest_loss.append(loss.item())  # each epoch loss

        if batch % 10 == 0: #For printing
                print(4*' ', '===> Test : [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    batch * len(Data), len(Test_loader.dataset),
                    100. * batch / len(Test_loader), np.mean(BatchVal_loss)))

        # for recreating the images
        for i in range (Att.shape[0]):

            Out_patchesTest.append(Att[i].cpu().detach().numpy())
            Target_Labels.append(Target[i].detach().numpy())
    # for saving test the images
    images = Reconstruct(np.asarray(Out_patchesTest),
                            Out_Patch_shape[0].shape, 14)
    images = Recreate_image(images, No_patch_shape[0].shape)
    images = MinMaxNorm(torch.from_numpy(images))

    images = images >= Threshold
    images = images.astype(int)
    for i in range(images.shape[0]):
        temp = np.asarray(images[i])
        temp = temp[0]
        im = Image.fromarray((temp*255).astype(np.uint8))
        im = im.save(f'Some_Save_Path/Test/{i+1}_Test.tif') #Add path

    return Train_loss, Val_loss, np.mean(BatchTest_loss)

#Loading data in.
DataDirPath = 'Some_Data_Path/MonuSeg' # Add path
DataPath = Get_Files(DataDirPath)

#Validation Data
Validation_Data = torch.load(DataPath[2])
X_Val, Y_Val = Validation_Data
X_Val = swapaxes(X_Val)
X_Val_Shape = X_Val[0].shape
Y_Val =  Y_Val.unsqueeze(1)
    # Needed for PRAUC
Patches_Val_Y = create_pathces(Y_Val, W, H, step_size = step_size)

# Test Data
#no need to scale test data. Only we use the attention mapping.
Test_Data = torch.load(DataPath[0])
X_test, Y_test = Test_Data
X_test = swapaxes(X_test)
Y_test =  Y_test.unsqueeze(1)

Patches_test_X = create_pathces(X_test, 1024, 1024, step_size = 1024)
X_Test_Data = Deconstruct(Patches_test_X)
Patches_test_Y = create_pathces(Y_test, 1024, 1024, step_size = 1024)
Y_Test_Data = Deconstruct(Patches_test_Y)
# ---------------------------------------------

Train_set = Models.DataSet(data_dir = 'Some_Data_Path/CostumDataSetTrain', transform = transform) #Add path
Train_loader = torch.utils.data.DataLoader(Train_set,
                        batch_size=batch_size,
                        shuffle=True)


Val_set = Models.DataSet(data_dir = 'Some_Data_Path/CostumDataSetVal', transform = transform ) #Add path
Val_loader = torch.utils.data.DataLoader(Val_set,
                        batch_size=batch_size,
                        shuffle=False)

#Test uses the normal dataloader
Test_set = torch.utils.data.TensorDataset(torch.tensor(X_Test_Data),
                                        torch.tensor(Y_Test_Data))

Test_loader = torch.utils.data.DataLoader(Test_set,
                        batch_size=2,
                        shuffle=False)

model = ChoseModel(num_S_Class, flag = True, num_class = num_class)

T_loss, V_loss, TesT_loss = Train_loop_SSL(model,
                      Train_loader,
                      Val_loader,
                      Test_loader,
                      epochs = epoch,
                      Loss_Fun = Loss_Fun,
                      Op = Optimizer, Lr = learning_rate,
                      Out_Patch_shape = Patches_test_X,
                      No_patch_shape = X_test,
                      Val_Patches = Patches_Val_Y,
                      Ori_Val_shape = X_Val_Shape,
                      Epsilon = Epsilon,
                      delta = Delta,
                      patience = Patience,
                      name = ChoseModel)

print(f'\nTest loss: {TesT_loss}')
print(f'Train loss : {T_loss}')

minpossTrain = T_loss.index(min(T_loss))
minpossVal = V_loss.index(min(V_loss))

plt.plot(range(len(T_loss)), T_loss, label = 'Train loss')
plt.plot(range(len(V_loss)), V_loss, label = 'Validation loss')
plt.axvline(minpossVal, linestyle='--', color='r',label='Early Stopping Checkpoint')
legend = plt.legend(loc = 'upper right', frameon = 1)
frame = legend.get_frame()
frame.set_facecolor('grey')
frame.set_edgecolor('black')
plt.xlabel ('Epoch')
plt.ylabel('Loss')
plt.title('Loss per epoch')
plt.savefig(f'/home/thire399/Documents/Bachelor_Project/Data/out/SSLUNet/MLP.png', dpi = 400)

plt.show()