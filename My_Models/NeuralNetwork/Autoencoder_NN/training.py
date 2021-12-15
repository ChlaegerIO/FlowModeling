# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
#from torch.autograd import Variable
#from torch.utils.data import DataLoader
#from torch.utils.data import Dataset
from torchvision import transforms
# read videos
from os import listdir
import cv2
# others
import os
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
        network['dx'], network['dx_sindy'], network['dz'], network['dz_sindy'], network['Xi'], network['rec_loss']
    params: hyperparameters
    separate_loss: dict of loss
        separate_loss['rec_loss'], separate_loss['sindy_x_loss'], separate_loss['sindy_z_loss'], separate_loss['sparse_loss']

    return summed_total_loss, separate_loss
    
    """
    dx = network['dx']
    dx_sindy = network['dx_sindy']
    dz = network['dz']
    dz_sindy = network['dz_sindy']
    Xi_coeff = network['Xi']
    rec_loss = network['rec_loss']
    sindy_x_loss = torch.mean((dx-dx_sindy)**2)
    sindy_z_loss = torch.mean((dz-dz_sindy)**2)
    sparse_loss = torch.mean(torch.abs(Xi_coeff))
    
    # separate view of each loss
    separate_loss = {}
    separate_loss['rec_loss'] = (float(rec_loss))
    separate_loss['sindy_x_loss'] = (float(sindy_x_loss))
    separate_loss['sindy_z_loss'] = (float(sindy_z_loss))
    separate_loss['sparse_loss'] = (float(sparse_loss))
    
    tot_loss = (params['loss_weight_decoder'] * rec_loss
                + params['loss_weight_sindy_x'] * sindy_x_loss 
                + params['loss_weight_sindy_z'] * sindy_z_loss
                + params['loss_weight_sindy_regularization'] * sparse_loss)
                                                                                        
    return tot_loss, separate_loss


def calculateSindy(network, params):
    '''
    Calculate Theta(z), Xi and dz
    
    '''
    z = network['z'].cpu().detach().numpy()
    dz = network['dz'].cpu().detach().numpy()
    
    network['Theta'] = torch.from_numpy(sindy.sindy_library(z, params['poly_order'], include_sine=params['include_sine']))
    network['Xi'] = torch.from_numpy(sindy.sindy_fit(network['Theta'], dz, params['sindy_threshold']))
    dz_predict = torch.matmul(network['Theta'],network['Xi']).cuda()
    
    return dz_predict



#############################################################################################################
# define model parameters
#############################################################################################################

params = {}

# autoencoder settings
params['number_epoch'] = 100                               # number of epochs
params['z_dim'] = 5                                     # number of coordinates for SINDy
params['batch_size'] = 16                                # batch size
params['lr_rate'] = 1e-5                                 # learning rate
params['weight_decay'] = 0


# loss function weighting
params['loss_weight_decoder'] = 1.0
params['loss_weight_sindy_x'] = 1e-4
params['loss_weight_sindy_z'] = 1e-10
params['loss_weight_sindy_regularization'] = 1e-4

# SINDy parameters
params['sindy_threshold'] = 0.5 
params['poly_order'] = 4
params['include_sine'] = False

# video processing
path_train = 'Videos/train/'
path_test = 'Videos/test/'
path_autoencoder = 'results/Autoencoder_#epoch_v1.pt'


#############################################################################################################
# data preprocessing
#############################################################################################################
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('cuda available: ', torch.cuda.is_available())
#print('cuda memory', torch.cuda.memory_summary(device=None, abbreviated=False))
torch.cuda.empty_cache()


# read the train videos in random order
file_names = []
for f in listdir(path_train):
    if f != 'high_res':
        file_names.append(f)

random.shuffle(file_names)

# define transform to tensor and resize to 1080x1920, 720x404 (16:9)
# pictures are 16:9 --> 1080x1920, 900x1600, 720x1280, 576x1024, 540x960: 500k pixel, 360x640, 272x480
# normalize = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])    # normalize around mean with sigma (std)
# transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((1080, 1920))])
transform = transforms.ToTensor()

# read data to list and transform to tensor
train_data_tmp = []
train_data = []
train_idxOfNewVideo = []
count = 0
for f in file_names:
    if count == 3:
        break
    count += 1
    train_idxOfNewVideo.append(len(train_data))
    vidcap = cv2.VideoCapture(path_train + f)
    success,imgR = vidcap.read()
    print('Read training data:',f)
    while success:
        imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)
        imgR_tensor = transform(imgR)
        train_data_tmp.append(imgR_tensor)
        success,imgR = vidcap.read()
        # make a batch
        if len(train_data_tmp) >= params['batch_size']:
            train_data.append(torch.stack(train_data_tmp))
            train_data_tmp = []
    
    print('train data: ', len(train_data), len(train_data[0]), len(train_data[0][0]), len(train_data[0][0][0]), len(train_data[0][0][0][0]))

print('index of new videos: ', train_idxOfNewVideo)
print('train data reading done!')

# split into validation and training set
validation_data = []
# take 10% of training set batches to validation set, take always two
nbr_batch = int(len(train_data)*0.1 / 2)
for i in range(0,nbr_batch):
    # choose position of train_idxOfNewVideo
    choose = random.randint(0, len(train_idxOfNewVideo)-1)
    whereInData = train_idxOfNewVideo[choose]
    # take first two frames of a video --> goal: no interruption of the video
    element1 = train_data[whereInData]
    element2 = train_data[whereInData+1]
    validation_data.append(element1)
    validation_data.append(element2)
    train_data.pop(whereInData+1)
    train_data.pop(whereInData)
    # adapt index where new videos start in train data
    for j in range(choose+1, len(train_idxOfNewVideo)):
        train_idxOfNewVideo[j] -= 2

print('validation data construction done: ', len(validation_data), len(validation_data[0]), len(validation_data[0][0]), len(validation_data[0][0][0]))
print('train data: ', len(train_data), len(train_data[0]), len(train_data[0][0]), len(train_data[0][0][0]), len(train_data[0][0][0][0]))
print('index of new videos: ', train_idxOfNewVideo)

# # read test videos
# test_data = []
# test_idxOfNewVideo = []

# # define transform to tensor and resize to 1080x1920, 720x404 (16:9)
# # normalize = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])    # normalize around mean with sigma (std)
# # transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((1080, 1920))])
# transform = transforms.ToTensor()

# # read data to list and transform to tensor
# count = 0
# for f in listdir(path_test):
#     if f != 'high_res':
#         count += 1
#         test_idxOfNewVideo.append(len(test_data))
#         vidcap = cv2.VideoCapture(path_test + f)
#         success,imgR = vidcap.read()
#         print('Read:',f)
#         while success:
#             imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)
#             imgR_tensor = transform(imgR)
#             test_data.append(imgR_tensor)
#             success,imgR = vidcap.read()
        
    
# print('test data reading done: ', len(test_data), len(test_data[0]), len(test_data[0][0]), len(test_data[0][0][0]))

#############################################################################################################
# autoencoder architecture
#############################################################################################################

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
            #print('train mode')
            encoded = self.encode(x)
            encoded = encoded.view(-1,200*8*10)
            encoded = self.fc1(encoded)

            decoded = self.fc2(encoded)
            decoded = decoded.view(-1,200,8,10)
            decoded = self.decode(decoded)
        else:
            #print('test mode')
            encoded = torch.zeros(1)

            decoded = self.fc2(z)
            decoded = decoded.view(-1,200,8,10)
            decoded = self.decode(decoded)
        
        return encoded, decoded

#############################################################################################################
# training loop
#############################################################################################################

# # load model
# if os.path.isfile(path_autoencoder):
#     autoencoder = torch.load(path_autoencoder)
#     print('loaded autoencoder', path_autoencoder)
# else:
#     autoencoder = Autoencoder()
#     print('loaded new autoencoder')


# to save network data
network = {}

# training function
def train(epoch):
    '''
    training function for the autoencoder: 
        use train_idxOfNewVideo to check if the current batch correspond to a new video

    epoch: current epoch of learning

    '''
    pos = 0
    tmp_save_loss = 0
    tmp_save_lossCategory = {}
    for batch_id, img_tensor in enumerate(train_data):
        img_tensor = img_tensor.cuda()
        encode_tensor, recon_tensor = autoencoder(img_tensor, 0, mode='train')
        network['rec_loss'] = criterion(recon_tensor, img_tensor)
    
        # x, z is current batch_id, dx, dz is next one (in else we take dz as current and compare with x from before, the excite to current step)
        if batch_id == train_idxOfNewVideo[pos]:
            pos += 1
            combined_loss = network['rec_loss']       
            network['z'] = encode_tensor#.float()
        else:
            network['dx'] = img_tensor#.float()
            network['dz'] = encode_tensor#.float()
            network['dz_sindy'] = calculateSindy(network, params).float()
            _, network['dx_sindy'] = autoencoder(0, network['dz_sindy'], mode='test')
            combined_loss, loss_category = calculateLoss(network, params)            # total loss with SINDy
            # now advance one step
            network['z'] = network['dz']
                
        # optimization and backpropagation
        optimizer.zero_grad()
        combined_loss.backward()
        optimizer.step()
        
        # print progress
        # printProgress(epoch, batch_id, combined_loss)
        writer.add_scalar('Training loss per batch', combined_loss, global_step=step_train)
        writer.add_image('clouds', img_tensor)
        writer.add_histogram('fc1', autoencoder.fc1.weight)
        tmp_save_loss = combined_loss
        tmp_save_lossCategory = loss_category
        step_train += 1
        img_tensor = img_tensor.detach()
    print('\n')

    # hyperparametering
    writer.add_hparams({'lr': lr_rate, 'zDimension': dimZ}, 
            {'combined loss': tmp_save_loss, 'sindy loss': tmp_save_lossCategory['sindy_x_loss'], 'sparsity loss': tmp_save_lossCategory['sparse_loss']})
    
    # delete from cuda
    # del encode_tensor
    # del recon_tensor


# evaluation function
evaluated_dict = {}
final_combined_loss = []
final_sindy_loss = []
def evaluate():
    '''
    evaluation of the training by it's loss

    '''
    autoencoder.eval()
    evaluated_combined_loss = 0
    evaluated_sindy_loss = 0
    for i, img_tensor in enumerate(validation_data):
        img_tensor = img_tensor.cuda()
        encode_eval_tensor, recon_eval_tensor = autoencoder(img_tensor, 0, mode='train')
        evaluated_dict['rec_loss'] = criterion(recon_eval_tensor, img_tensor)
        print('i: ', i)

        # first batch
        if i % 2 == 0:
            # x, z are at the current time
            evaluated_dict['z'] = encode_eval_tensor#.float()
            
        # second batch, evaluation
        else:
            evaluated_dict['dx'] = img_tensor#.float()
            evaluated_dict['dz'] = encode_eval_tensor#.float()
            eval_theta = torch.from_numpy(sindy.sindy_library(evaluated_dict['z'].cpu().detach().numpy(), params['poly_order'], include_sine=params['include_sine']))
            evaluated_dict['dz_sindy'] = torch.matmul(eval_theta,network['Xi']).float().cuda()
            
            evaluated_dict['Xi'] = network['Xi']
            _, evaluated_dict['dx_sindy'] = autoencoder(0, evaluated_dict['dz_sindy'], mode='test')
            combined_loss, loss_category = calculateLoss(evaluated_dict, params)            # total loss with SINDy

            # loss
            evaluated_combined_loss += combined_loss
            evaluated_sindy_loss += loss_category['sindy_x_loss']
    
    # append average loss of this epoch
    evaluated_combined_loss_perData = evaluated_combined_loss/len(validation_data)*2
    evaluated_sindy_loss_perData = evaluated_sindy_loss/len(validation_data)*2
    final_combined_loss.append(evaluated_combined_loss_perData)
    final_sindy_loss.append(evaluated_sindy_loss_perData)
    writer.add_scalar('Evaluation combined loss per epoch', evaluated_combined_loss_perData, global_step=step_eval)
    writer.add_scalar('Evaluation sindy loss per epoch', evaluated_sindy_loss_perData, global_step=step_eval)
    step_eval += 1

    # del encode_eval_tensor
    # del recon_eval_tensor
    # del recon_sindy_eval_tensor



# print to tensorboard and hyperparameter search
lr_rate_arr = [0.01, 0.001, 0.0001, 0.00001]
dim_z_arr = [1, 2, 3, 10, 100, 1000]

for lr_rate in lr_rate_arr:
    params['lr_rate'] = lr_rate
    for dimZ in dim_z_arr:
        params['z_dim'] = dimZ
        writer = SummaryWriter(f'runs/v4Tensorboard/trainLoss_LR{lr_rate}_dimZ{dimZ}')
        step_train = 0
        step_eval = 0

        # load new network
        autoencoder = Autoencoder()
        autoencoder = autoencoder.cuda()
        # optimization technique
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=params['lr_rate'], weight_decay=params['weight_decay'])

        # epoch loop
        for epoch in range(params['number_epoch']):
            train(epoch)
            print('train epoch', epoch, 'done')
            evaluate()
            print('evaluate epoch', epoch, 'done')

            # save model every 100 epoch
            if epoch % 50 == 0:
                name_Ae = 'results/Ae_' + str(epoch) + 'epoch_bs16_lr{lr_rate}_z{dimZ}_sindth0-5_poly5.pt'
                name_Xi = 'results/Xi_' + str(epoch) + 'epoch_bs16_lr{lr_rate}_z{dimZ}_sindth0-5_poly5.pt'
                torch.save(autoencoder, name_Ae)
                torch.save(network['Xi'], name_Xi)
                print('saved model in epoch', epoch)

    torch.cuda.empty_cache()
