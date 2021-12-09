# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.autograd import Variable
#from torch.utils.data import DataLoader
#from torch.utils.data import Dataset
from torchvision import transforms
# read videos
#import pylab          # play video
from os import listdir
import cv2
# others
import os
#from PIL import Image
import random
#import matplotlib.pyplot as plt
# SINDy
import sys
sys.path.append("src")
import sindy_utils as sindy
import numpy as np

#############################################################################################################
# helping function implementation
#############################################################################################################

print('hello its me')

def printProgress(epoch, batch_id, loss):
    """
    print progress of the training
    epoch: number
    batch_id: current batch_id
    accuracy:
    loss:
    
    """
    progress = '='* int((10. * (batch_id+1) / len(train_data)))
    progress += '>'
    if batch_id == 0:
        print('Train Epoche {}: {}% [{}]\t , loss: {:.6f}'.format(
            epoch+1, int(100. * (batch_id+1) / len(train_data)),progress, loss.item()), end='')
    else:
        print('\rTrain Epoche {}: {}% [{}]\t , loss: {:.6f}'.format(
            epoch+1, int(100. * (batch_id+1) / len(train_data)),progress, loss.item()), end='', flush = True)
        

def calculateLoss(network, params):
    """
    calculate the loss of autoencoder and SINDy combined. loss function of:
    
     O \   _________           ________  /  O
     .    |         | \  O  / |        |    .
     . -  | phi'(x) | -  O  - | phi(z) | -  .
     .    |_________| /  O  \ |________|    .
     O /                                 \  O
    x(t)                z(t)               xa(t)
    
    ||x-phi(z)||_2^2 + lam1 ||dx - (zGrad phi(z)) Theta(z^T) Xi||_2^2 + lam2 ||dz - Theta(z^T) Xi||_2^2 + lam3 ||Xi||_1
        decoder      +                   SINDy in dx                  +         SINDy in dz             +   SINDy sparsity
     
    dz = xGrad phi'(x) dx = (xGrad z) dx
    
    network: data of the network
    params: hyperparameters
    
    """
    dx = network['dx']
    dx_decode = network['dx_decode']
    dz = network['dz']
    dz_predict = network['dz_sindy']
    Xi_coeff = network['Xi']
    rec_loss = network['rec_loss']
    sindy_x_loss = torch.mean((dx-dx_decode)**2)
    sindy_z_loss = torch.mean((dz-dz_predict)**2)
    sparse_loss = torch.mean(torch.abs(Xi_coeff))
    
    # separate view of each loss
    separate_loss = []
    separate_loss.append(float(rec_loss))
    separate_loss.append(float(sindy_x_loss))
    separate_loss.append(float(sindy_z_loss))
    separate_loss.append(float(sparse_loss))
    
    tot_loss = (params['loss_weight_decoder'] * rec_loss
                + params['loss_weight_sindy_x'] * sindy_x_loss 
                + params['loss_weight_sindy_z'] * sindy_z_loss
                + params['loss_weight_sindy_regularization'] * sparse_loss)
                                                                                        
    return tot_loss, separate_loss


def calculateSindy(network, params):
    '''
    Calculate Theta(z), Xi and dz
    
    '''
    z = network['z'].detach().numpy()
    dz = network['dz'].detach().numpy()
    
    network['Theta'] = torch.from_numpy(sindy.sindy_library(z, params['poly_order'], include_sine=params['include_sine']))
    network['Xi'] = torch.from_numpy(sindy.sindy_fit(network['Theta'], dz, params['sindy_threshold']))
    dz_predict = torch.matmul(network['Theta'],network['Xi'])
    
    return dz_predict



#############################################################################################################
# define model parameters
#############################################################################################################

params = {}

# autoencoder settings
params['number_epoch'] = 10000                               # number of epochs
params['z_dim'] = 5                                     # number of coordinates for SINDy
params['batch_size'] = 32                                # batch size
params['lr_rate'] = 0.1                                 # learning rate
params['weight_decay'] = 1e-8


# loss function weighting
params['loss_weight_decoder'] = 1.0
params['loss_weight_sindy_x'] = 1e-4
params['loss_weight_sindy_z'] = 1e-10
params['loss_weight_sindy_regularization'] = 1e-4

# SINDy parameters
params['sindy_threshold'] = 0.5 
params['poly_order'] = 4
params['include_sine'] = False


#############################################################################################################
# data preprocessing
#############################################################################################################
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('cuda available: ', torch.cuda.is_available())
#print('cuda memory', torch.cuda.memory_summary(device=None, abbreviated=False))
torch.cuda.empty_cache()


# read the train videos in random order
file_names = []
for f in listdir('Videos/train/'):
    if f != 'high_res':
        file_names.append(f)

random.shuffle(file_names)

# define transform to tensor and resize to 1080x1920
normalize = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])    # normalize around mean with sigma (std)
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((1080, 1920))])

# read data to list and transform to tensor
train_data_tmp = []
train_data = []
train_idxOfNewVideo = []
count = 0
for f in file_names:
    # just for testing (save time)
    # if count == 1:
    #     break
    # job gets killed with to much data?
    #if len(train_data) >= 60:
    #    break
    count += 1
    vidcap = cv2.VideoCapture('Videos/train/' + f)
    success,imgR = vidcap.read()
    print('Read training data:',f)
    while success:
        imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)
        imgR_tensor = transform(imgR)
        train_data_tmp.append(imgR_tensor)
        success,imgR = vidcap.read()
        #if len(train_data) >= 60:
        #    break
        # make a batch
        if len(train_data_tmp) >= params['batch_size']:
            train_data.append(torch.stack(train_data_tmp))
            train_data_tmp = []
    train_idxOfNewVideo.append(len(train_data))
    print('train data: ', len(train_data), len(train_data[0]), len(train_data[0][0]), len(train_data[0][0][0]), len(train_data[0][0][0][0]))



# split into validation and training set
validation_data = []
# take 10% of training set batches to validation set
nbr_batch = int(len(train_data)*0.1)
for i in range(0,nbr_batch):
    choose = random.randint(0, len(train_data)-1)
    element = train_data[choose]
    validation_data.append(element)
    train_data.pop(choose)

print('validation data: ', len(validation_data), len(validation_data[0]), len(validation_data[0][0]), len(validation_data[0][0][0]))
print('train data: ', len(train_data), len(train_data[0]), len(train_data[0][0]), len(train_data[0][0][0]), len(train_data[0][0][0][0]))


# read test videos
test_data = []
test_idxOfNewVideo = []

# read data to list and transform to tensor
count = 0
# for f in listdir('Videos/test/'):
#     if f != 'high_res':
#         # just for testing (save time)
#         # if count == 1:
#         #     break
#         count += 1
#         vidcap = cv2.VideoCapture('Videos/test/' + f)
#         success,imgR = vidcap.read()
#         print('Read:',f)
#         while success:
#             imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)
#             imgR_tensor = transform(imgR)
#             test_data.append(imgR_tensor)
#             success,imgR = vidcap.read()
#         test_idxOfNewVideo.append(len(test_data))
    
# print('test data: ', len(test_data), len(test_data[0]), len(test_data[0][0]), len(test_data[0][0][0]))


#############################################################################################################
# autoencoder architecture
#############################################################################################################

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__() 
        self.encode = nn.Sequential(
            # encoder: N, 3, 1080, 1920
            nn.Conv2d(3, 16, 5), # N, 16, 1076, 1916
            nn.ReLU(),
            nn.Conv2d(16, 32, 5), # N, 32, 1072, 1912
            nn.ReLU(),
            nn.MaxPool2d((2,3), stride=(2,3)), # N, 32, 536, 637              -- pool --
            nn.Conv2d(32, 64, 8, stride=2, padding=1), # N, 64, 266, 316
            nn.ReLU(),
            nn.Conv2d(64, 96, 7), # N, 96, 260, 310
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2), # N, 96, 130, 155                      -- pool --
            nn.Conv2d(96, 128, 11), # N, 128, 120, 145
            nn.ReLU(),
            nn.Conv2d(128, 150, 4, stride=2, padding=1), # N, 150, 60, 72
            nn.ReLU(),
            nn.Conv2d(150, 200, 11), # N, 200, 50, 62
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2), # N, 200, 25, 31                        -- pool --
            nn.Conv2d(200, 200, 9, stride=2, padding=1), # N, 200, 10, 13
            nn.ReLU(),
            nn.Conv2d(200, 200, 10), # N, 200, 1, 4
            nn.ReLU()
        )
        
        self.fc1 = nn.Linear(200*4,params['z_dim'])
        # Note: nn.MaxPool2d -> use nn.MaxUnpool2d, or use different kernelsize, stride etc to compensate...
        # Input [-1, +1] -> use nn.Tanh    
        
        # note: encoder and decoder are not symmetric
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(200, 200, 4), # N, 200, 4, 7
            nn.ReLU(),
            nn.ConvTranspose2d(200, 200, 9, stride=2, padding=2, output_padding=(1,0)), # N, 200, 12, 17
            nn.ReLU(),
            nn.ConvTranspose2d(200, 150, 8, stride=4, padding=(0,1), output_padding=(3,0)), # N, 150, 55, 70
            nn.ReLU(),
            nn.ConvTranspose2d(150, 128, 8), # N, 128, 62, 77
            nn.ReLU(),
            nn.ConvTranspose2d(128, 96, 6, stride=(2,3), padding=1), # N, 96, 126, 232
            nn.ReLU(),
            nn.ConvTranspose2d(96, 64, 7), # N, 64, 132, 238
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 8, stride=2, padding=(1,2), output_padding=(1,1)), # N, 32, 296, 479
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 5, stride=2, padding=1), # N, 16, 539, 959
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 5, stride=2, padding=1, output_padding=(1,1)), # N, 3, 1080, 1920
            nn.ReLU()
        )   
        
        self.fc2 = nn.Linear(params['z_dim'], 200*4)

    def forward(self, x, z, mode):
        '''
        x: input for encoder
        z: input for decoder
        mode: 
            'train' -> use encoded for decoder
            'test'  -> feed z in an get decoded
        
        '''
        if mode == 'train':
            #print('train mode')
            encoded = self.encode(x)
            encoded = encoded.view(-1,200*4)
            encoded = self.fc1(encoded)

            decoded = self.fc2(encoded)
            decoded = decoded.view(-1,200,1,4)
            decoded = self.decode(decoded)            
        else:
            #print('test mode')
            encoded = torch.zeros(1)

            decoded = self.fc2(z)
            decoded = decoded.view(-1,200,1,4)
            decoded = self.decode(decoded)
        
        return encoded, decoded

#############################################################################################################
# training loop
#############################################################################################################

# load model
autoencoder_path = 'results/Autoencoder_#epoch_v1.pt'
if os.path.isfile(autoencoder_path):
    autoencoder = torch.load(autoencoder_path)
    print('loaded autoencoder', autoencoder_path)
else:
    autoencoder = Autoencoder()
    print('loaded new autoencoder')
autoencoder = autoencoder#.cuda()

# optimization technique
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=params['lr_rate'], weight_decay=params['weight_decay'])

# to save network data
network = {}

# training function
outputs = []
def train(epoch):
    '''
    training function for the autoencoder

    '''
    for batch_id,img in enumerate(train_data):
        img = img.cuda()  
        encode_tensor, recon_tensor = autoencoder(img, 0, mode='train')
        network['rec_loss'] = criterion(recon_tensor, img)
    
        # x, z is current batch_id, dx, dz is next one (in else we take dz as current and compare with x from before)
        if batch_id == 0:
            combined_loss = network['rec_loss']       
            network['z'] = float(encode_tensor)
        else:
            network['dx'] = float(img)
            network['dz'] = float(encode_tensor)
            network['dz_sindy'] = calculateSindy(network, params).float()
            _, network['dx_decode'] = autoencoder(0, network['dz_sindy'], mode='test')
            combined_loss, loss_category = calculateLoss(network, params)            # total loss with SINDy
            # now advance one step
            network['z'] = network['dz']
                
        # optimization and backpropagation
        optimizer.zero_grad()
        combined_loss.backward()
        optimizer.step()
        
        # print progress
        printProgress(epoch, batch_id, combined_loss)
        img = img.detach()
        # if batch_id == 2:
        #     break
    print('\n')
    outputs.append((epoch, float(img), float(recon_tensor), float(combined_loss), loss_category))

    
# evaluation function
evaluated_dict = {}
ae_loss = []
sindy_loss = []
def evaluate():
    '''
    evaluation of the training by it's loss

    '''
    autoencoder.eval()
    ae_lossE = 0
    sindy_lossE = 0
    for i, img in enumerate(validation_data):
        # an other video sequence
        if i % params['batch_size'] == 0:
            # x, z are at the current time
            encode_eval_tensor, recon_eval_tensor = autoencoder(img, 0, mode='train')
            evaluated_dict['x'] = float(recon_eval_tensor)
            evaluated_dict['z'] = float(encode_eval_tensor)
            eval_theta = torch.from_numpy(sindy.sindy_library(evaluated_dict['z'], params['poly_order'], include_sine=params['include_sine']))
            evaluated_dict['dz_sindy'] = torch.matmul(eval_theta,network['Xi']).float()
            _, evaluated_dict['dx_sindy'] = autoencoder(0, evaluated_dict['dz_sindy'], mode='test')
            # autoencoder loss
            ae_lossE += float(F.mse_loss(evaluated_dict['x'], img))
        else:
            # sindy loss
            sindy_lossE += float(F.mse_loss(evaluated_dict['dx_sindy'], img))
            evaluated_dict['z'], evaluated_dict['x'] = autoencoder(img, 0, mode='train')            
            eval_theta = torch.from_numpy(sindy.sindy_library(evaluated_dict['z'].detach().numpy(), params['poly_order'], include_sine=params['include_sine']))
            evaluated_dict['dz_sindy'] = torch.matmul(eval_theta,network['Xi']).float()
            _, recon_sindy_eval_tensor = autoencoder(0, evaluated_dict['dz_sindy'], mode='test')
            evaluated_dict['dx_sindy'] = float(recon_sindy_eval_tensor)
            ae_lossE += float(F.mse_loss(evaluated_dict['x'], img))
    
    # append average loss of this epoch
    ae_loss.append(ae_lossE/len(validation_data))
    sindy_loss.append(sindy_lossE/len(validation_data))

# epoch loop
for epoch in range(params['number_epoch']):
    train(epoch)
    evaluate()



# save model
torch.save(autoencoder, 'results/Ae_10000epoch_bs64_lr0-1_z5_sindt0-5_poly5.pt')
torch.save(autoencoder, 'results/Xi_10000epoch_bs64_lr0-1_z5_sindt0-5_poly5.pt')

torch.cuda.empty_cache()
