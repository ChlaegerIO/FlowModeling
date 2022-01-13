import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
import cv2
import random
import sys
sys.path.append("src")
import sindy_utils as sindy

# autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__() 
        self.encode = nn.Sequential(
            # encoder: N, 3, 404, 720
            nn.Conv2d(3, 16, 2), # N, 16, 403, 719
            nn.ReLU(),
            nn.Conv2d(16, 32, 2), # N, 32, 402, 718
            nn.ReLU(),
            nn.MaxPool2d((2,3), stride=(2,3)), # N, 32, 201, 239              -- pool --
            nn.Conv2d(32, 64, 4), # N, 64, 198, 236
            nn.ReLU(),
            nn.Conv2d(64, 96, 4), # N, 96, 195, 233
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2), # N, 96, 97, 116                       -- pool --
            nn.Conv2d(96, 128, 5), # N, 128, 93, 112
            nn.ReLU(),
            nn.Conv2d(128, 150, 5, stride=2, padding=1), # N, 150, 46, 55
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2), # N, 150, 23, 27                        -- pool --
            nn.Conv2d(150, 200, 9, stride=2), # N, 200, 8, 10
            nn.ReLU()
        )
        
        self.fc1 = nn.Linear(200*8*10,params['z_dim'])
        # Note: nn.MaxPool2d -> use nn.MaxUnpool2d, or use different kernelsize, stride etc to compensate...
        # Input [-1, +1] -> use nn.Tanh    
        
        # note: encoder and decoder are not symmetric
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(200, 150, 4), # N, 150, 11, 13
            nn.ReLU(),
            nn.ConvTranspose2d(150, 128, 5, stride=(2,3), padding=(2,2), output_padding=(0,2)), # N, 128, 21, 39
            nn.ReLU(),
            nn.ConvTranspose2d(128, 96, 4, stride=2, padding=(1,0)), # N, 96, 42, 80
            nn.ReLU(),
            nn.ConvTranspose2d(96, 64, 8), # N, 64, 49, 87
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 8, stride=2, padding=(2,1), output_padding=(0,1)), # N, 32, 100, 179
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 5, stride=2, padding=1), # N, 16, 201, 359
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 5, stride=2, padding=1, output_padding=(1,1)), # N, 3, 404, 720
            nn.ReLU()
        )   
        
        self.fc2 = nn.Linear(params['z_dim'], 200*8*10)

    def forward(self, x, z, mode):
        '''
        x: input for encoder
        z: input for decoder
        mode: 
            'train' -> use encoded for decoder
            'test'  -> feed z in an get decoded
        
        '''
        if mode == 'train':
            encoded = self.encode(x)
            encoded = encoded.view(-1,200*8*10)
            encoded = self.fc1(encoded)

            decoded = self.fc2(encoded)
            decoded = decoded.view(-1,200,8,10)
            decoded = self.decode(decoded)
        else:
            encoded = torch.zeros(1)

            decoded = self.fc2(z)
            decoded = decoded.view(-1,200,8,10)
            decoded = self.decode(decoded)
        
        return encoded, decoded

    
def calculateSindy(z, Xi, poly_order, include_sine_param):
    z_new = z.detach().numpy()
    
    theta = torch.from_numpy(sindy.sindy_library(z_new, poly_order, include_sine=include_sine_param))
    
    dz_prediction = torch.matmul(theta, Xi).float()
    
    return dz_prediction


# loading model
path_folder = 'results/v5/'

to_load = path_folder+'Ae_4000epoch_bs16_lr1e-5_z2_sindt05_poly5.pt'
autoencoder = torch.load(to_load)
autoencoder = autoencoder.cpu()

# load a train data
path_folder_data = 'results/v5/data/'
train_data = torch.load(path_folder_data + 'train_data.pt')
print('train data: ', len(train_data), len(train_data[0]), len(train_data[0][0]), len(train_data[0][0][0]), len(train_data[0][0][0][0]))
print('train data reading done!')

## load a validation data
#validation_data = torch.load(path_folder_data + 'validation_data.pt')
#print('validation data: ', len(validation_data), len(validation_data[0]), len(validation_data[0][0]), len(validation_data[0][0][0]), len(validation_data[0][0][0][0]))
#print('validation data reading done!')
#
## loading test data
#test_data = torch.load(path_folder_data + 'test_data.pt')
#print('test data: ', len(test_data), len(test_data[0]), len(test_data[0][0]), len(test_data[0][0][0]), len(test_data[0][0][0][0]))
#print('test data reading done!')

def matrixToNorm(x, offset=0, factor=0.95):
    x = (x - x.min() + offset) / x.max() * factor
    return x

poly_order = 4
include_sine_param = False
threshold_sindy = 0.1
until = 5                      # choose the number of prediction stepts, 1 step are number of batch_size frames


# use more than 16 frames to get Xi
def constructXi(data, zDim):
    '''
    input: data as a list with shape [len batch_size RGB hight width]
    return: Xi
    
    '''
    # processs the data
    z_tensor = torch.empty((0, zDim))
    data_len = len(data)
    for i in range(data_len):
        z_tensor_tmp, _ = autoencoder(train_data[i], 0, mode='train')
        z_tensor = torch.cat((z_tensor, z_tensor_tmp), 0)
        if i % 5 == 0:
            print(i, z_tensor.shape)
        del z_tensor_tmp

    print(z_tensor.shape)
    
    dz_tensor = z_tensor[2:data_len]
    z_tensor = z_tensor[1:data_len-1]
    
    # calculate sindy and Xi for the data
    z = z_tensor.cpu().detach().numpy()
    dz = dz_tensor.cpu().detach().numpy()

    Theta = torch.from_numpy(sindy.sindy_library(z, poly_order, include_sine=include_sine_param))
    Xi = torch.from_numpy(sindy.sindy_fit(Theta, dz, threshold_sindy))
    
    return Xi


Xi = constructXi(train_data, zDim=2)

print(Xi)
lr_rate = 1e-5
dimZ = 2
epoch = 3000
name_Xi = 'results/v5_3/Xi_' + str(epoch) + 'epoch_bs16_lr' + str(lr_rate) + '_z' + str(dimZ) + '_sindt05_poly5.pt'
torch.save(Xi, name_Xi)