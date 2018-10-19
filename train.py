import os
import random
import math
import numpy as np
import torch
import torch.nn as nn             # neural network lib.
import torch.nn.functional as F   # common functions (e.g. relu, drop-out, softmax...)
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models




# Disable this variable if running in a CUDA disabled computer
USE_GPU = True
# Enable this variable if to train the model
WILL_TRAIN = False
# Enable this variable if to test the trained model
WILL_TEST = True
# Use 6 workers if CUBA enabled, otherwise 0 workers
num_of_workers = 0
if USE_GPU:
    num_of_workers = 6
    torch.set_default_tensor_type('torch.cuda.FloatTensor')




def readTXTFile(file_path):
    '''
        Function to read a txt file containing key information of faces
        :param file_path [str] - path to input txt file
        :return [list] - list of input data
    '''
    return_list = []
    with open(file_path, "r") as f:
        for line in f:
            tokens = line.split('\t') # split the line by tab, 3 items in total
            if len(tokens) < 3:
                continue
            img_name = str(tokens[0])
            img_crop = list(map(float, tokens[1].split())) # tokens[1].split()
            landmark = list(map(float, tokens[2].split()))
            return_list.append({
                'img_name': img_name,
                'file_path': os.path.join(img_name[:-9], img_name),
                'img_crop': img_crop,
                'landmark': landmark
            })
    return return_list


def drawImageAndCircles(img_face, img_landmark):
    '''
        Function to draw image and landmarks via cv2
        :param img_face [arr] - np array reFalsepresents the image of a face
        :param img_landmark [arr] - np array of coordinates represents the landmarks
        :return - Noneexchange
    '''
    ok_flag = True
    img_landmark_points = zip(img_landmark[::2], img_landmark[1::2])
    while ok_flag:
        for point in img_landmark_points:
            # print(point)
            cv2.circle(img_face, (int(point[0]), int(point[1])), 5, (255, 0, 0))
        img_not_blue = img_face.copy()
        print(img_face)
        img_not_blue[:, :, 0] = img_face[:, :, 2]
        img_not_blue[:, :, 2] = img_face[:, :, 0]
        cv2.imshow('image', img_not_blue)
        if cv2.waitKey(0) == 27:
            ok_flag = False
    cv2.destroyAllWindows()


def drawImageAndCirclesWithPLT(image, landmark_norm):
    '''
        Function to draw image and landmarks via PLT
        :param img_face [torch image] - np array represents the image of a face
        :param landmark_norm [torch arr] - np array of coordinates represents the landmarks, normalized
        :return - None
    '''
    np_img = image.cpu().numpy().squeeze()
    print(landmark_norm)
    np_landmark = landmark_norm.detach().numpy().squeeze() * 225
    plt.imshow(np_img.transpose((2,1,0)).copy())
    plt.scatter(np_landmark[0::2], np_landmark[1::2], 50, 'r', '.')
    plt.show()



class LFWDataset(Dataset):
    '''
        Class to represent face image dataset, extend dataset class
        :func init (data_list) - init LFWDataset
        :func len - get length of dataset
        :func getitem (idx) - get one item of the dataset
    '''
    def __init__(self, data_list):
        '''
            Function to load the dataset list
            :param data_list [arr] - list of data
        '''
        self.data_list = data_list
        self.lfw_dataset_dir = 'lfw'

    def __len__(self):
        '''
            Function to get length of the dataset list
        '''
        return len(self.data_list)

    def __getitem__(self, idx):
        '''
            Function to get one item from the dataset list
            :param idx [int] - index of item to get
        '''
        item = self.data_list[idx]
        img_name = np.asarray(item['img_name'])
        img_landmark = np.asarray(item['landmark'])
        file_path = os.path.join(self.lfw_dataset_dir, item['file_path'])
        # Helper function for random crop and random flip
        will_flip = random.uniform(0, 1)
        will_ran_crop = random.uniform(0, 1)
        if will_ran_crop > 0.5:
            delta_x = min(img_landmark[0] - item['img_crop'][0], item['img_crop'][2] - img_landmark[6]) * random.uniform(-0.5, 0.5)
            delta_y = min(img_landmark[1] - item['img_crop'][1], img_landmark[7] - item['img_crop'][1]) * random.uniform(0, 0.5)
            item['img_crop'][0::2] = item['img_crop'][0::2] + delta_x
            item['img_crop'][1::2] = item['img_crop'][1::2] + delta_y
        # Crop img, relabel label
        # Resize coordinates of img and label to 225 * 225
        img_original = Image.open(file_path)
        img_face = np.asarray(img_original.crop(tuple(item['img_crop'])), dtype=np.float32) / 255.0 * 2 - 1
        img_cropped_h,img_cropped_w = img_face.shape[0], img_face.shape[1]
        resized_w = resized_h = 225
        resized_img_face = cv2.resize(img_face, (resized_w, resized_h))
        if will_flip > 0.5:
            resized_img_face = cv2.flip(resized_img_face, 1)
        # Crop img, relabel label
        # Resize coordinates of img and label to 225 * 225
        # Normalize rgb to 0 - 1
        if will_flip > 0.5:
            img_landmark[0::2] = item['img_crop'][2] - (img_landmark[0::2] - item['img_crop'][0])  # x coordinates after flip
            img_landmark_temp = img_landmark[0:1:1]
            img_landmark[0],img_landmark[6]=img_landmark[6],img_landmark[0]
            img_landmark[1],img_landmark[7]=img_landmark[7],img_landmark[1]
            img_landmark[2],img_landmark[4]=img_landmark[4],img_landmark[2]
            img_landmark[3],img_landmark[5]=img_landmark[5],img_landmark[3]
            img_landmark[8],img_landmark[10]=img_landmark[10],img_landmark[8]
            img_landmark[9],img_landmark[11]=img_landmark[11],img_landmark[9]
        img_landmark[0::2] = (img_landmark[0::2] - item['img_crop'][0]) / img_cropped_w * resized_w # x coordinates
        img_landmark[1::2] = (img_landmark[1::2] - item['img_crop'][1]) / img_cropped_h * resized_h # y coordinates
        img_landmark_norm = img_landmark / 225
        # # Create image tensor
        img_tensor = torch.from_numpy(resized_img_face)
        # Reshape to (1, 28, 28), the 1 is the channel size
        img_tensor = torch.transpose(img_tensor, 0, 2)
        landmark_tensor = torch.from_numpy(img_landmark_norm).float()      # Loss measure require long type tensor
        return img_tensor, landmark_tensor




## Read all data in txt files
label_file_path = os.path.join('LFW_annotation_train.txt')
test_file_path = os.path.join('LFW_annotation_test.txt')
label_list = readTXTFile(label_file_path)
test_list = readTXTFile(test_file_path)
random.shuffle(label_list)


## Divide data into train, validate and test lists
total_label_items = len(label_list)
total_test_items = len(test_list)
n_train_sets = total_label_items * .7
train_set_list = label_list[: int(n_train_sets)]
n_valid_sets = total_label_items * .3
valid_set_list = label_list[int(n_train_sets): int(n_train_sets + n_valid_sets)]
test_set_list = test_list


## Load train and validate data into datasets
train_dataset = LFWDataset(train_set_list)
train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=128,
                                                shuffle=True,
                                                num_workers=num_of_workers)
print('Total training items', len(train_dataset), ', Total training mini-batches in one epoch:', len(train_data_loader))
valid_dataset = LFWDataset(valid_set_list)
valid_data_loader = torch.utils.data.DataLoader(valid_dataset,
                                                batch_size=32,
                                                shuffle=True,
                                                num_workers=num_of_workers)
print('Total validation set:', len(valid_dataset))
test_dataset = LFWDataset(test_set_list)
test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=1,
                                                shuffle=True,
                                                num_workers=num_of_workers)
print('Total test set:', len(test_dataset))




# Apply and finetuning the Alexnet
if WILL_TRAIN:
    ## Print key info of img and landmark inputs
    idx, (image, landmark) = next(enumerate(train_data_loader))  # we can use next(*) load once.
    print('image tensor shape (N, C, H, W):', image.shape)
    print('landmark tensor shape (N, landmarks):', landmark.shape)

    net = models.alexnet(pretrained=True)
    net.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 14),
        )
    if USE_GPU:
        net.cuda()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = 0.0001)

    # losses collection, used for monitoring over-fit
    train_losses = []
    valid_losses = []
    max_epochs = 6
    itr = 0

    for epoch_idx in range(0, max_epochs):
        for train_batch_idx, (train_input, train_label) in enumerate(train_data_loader):
            itr += 1
            net.train() # switch to train model
            optimizer.zero_grad() # zero the parameter gradients
            # Forward
            train_input = Variable(train_input.cuda())          # use Variable(*) to allow gradient flow
            train_out = net.forward(train_input)                # forward once
            # compute loss
            train_label = Variable(train_label.cuda())
            loss = criterion(train_out, train_label)
            # do the backward and compute gradients
            loss.backward()
            # update the parameters with SGD
            optimizer.step()
            # Add the tuple of （iteration, loss) into `train_losses` list
            train_losses.append((itr, loss.item()))
            # Run the validation every 200 iteration:
            if train_batch_idx % 20 == 0:
                print('Epoch: %d Itr: %d Loss: %f' % (epoch_idx, itr, loss.item()))
            if train_batch_idx % 20 == 0:
                net.eval()             # [Important!] set the network in evaluation net
                valid_loss_set = []    # collect the validation losses
                valid_itr = 0
                # Do validation
                for valid_batch_idx, (valid_input, valid_label) in enumerate(valid_data_loader):
                    net.eval()
                    valid_input = Variable(valid_input.cuda())  # use Variable(*) to allow gradient flow
                    valid_out = net.forward(valid_input)  # forward once
                    # compute loss
                    valid_label = Variable(valid_label.cuda())
                    valid_loss = criterion(valid_out, valid_label)
                    valid_loss_set.append(valid_loss.item())

                    # We just need to test 5 validation mini-batchs
                    valid_itr += 1
                    if valid_itr > 5:
                        break
                # Compute the avg. validation loss
                avg_valid_loss = np.mean(np.asarray(valid_loss_set))
                print('Valid Epoch: %d Itr: %d Loss: %f' % (epoch_idx, itr, avg_valid_loss))
                valid_losses.append((itr, avg_valid_loss))

    train_losses = np.asarray(train_losses)
    valid_losses = np.asarray(valid_losses)

    plt.plot(train_losses[:, 0],      # iteration
             train_losses[:, 1])      # loss value
    plt.plot(valid_losses[:, 0],      # iteration
             valid_losses[:, 1])      # loss value
    plt.xlabel('Iteration')
    plt.ylabel('Losses')
    plt.legend(['Train losses','Valid losses'])
    plt.gca().set_yticklabels(['{:.2f}'.format(x) for x in plt.gca().get_yticks()])
    plt.show()

    # save model to disk
    net_state = net.state_dict()                                             # serialize trained model
    torch.save(net_state, 'lfw_net.pth')    # save to disk




if WILL_TEST:
    ## Test instance
    test_count = 0
    test_limit = 2007
    x_axis = np.linspace(0, 40, 41)
    y_axis = np.zeros(41)
    offset_all = []
    test_net = models.alexnet(pretrained=True) # Create a net instance
    test_net.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 14),
        )
    if USE_GPU:
        test_net_state = torch.load('lfw_net.pth') # Load serialize data
    else:
        test_net_state = torch.load('lfw_net.pth', map_location='cpu') # Load serialize data
    test_net.load_state_dict(test_net_state) # Load weights/parameters from serialized data
    test_net.eval() # Set the network to evaluate model

    for test_idx, (test_image, test_landmark) in enumerate(test_data_loader):
        if USE_GPU:
            test_image = Variable(test_image.cuda())  # use Variable(*) to allow gradient flow
            test_out = test_net.forward(test_image.cuda())  # forward once
            test_landmark = Variable(test_landmark.cuda())
        else:
            test_out = test_net.forward(test_image)  # forward once

        if USE_GPU:
            np_label = test_landmark.cpu().numpy().squeeze() * 225
            np_out = test_out.cpu().numpy().squeeze() * 225
        else:
            np_label = test_landmark.detach().numpy().squeeze() * 225
            np_out = test_out.detach().numpy().squeeze() * 225
        offset = np.zeros(np_out.shape[0])
        offset[0::2] = [i ** 2 for i in (np_out[0::2] - np_label[0::2])]
        offset[1::2] = [i ** 2 for i in (np_out[1::2] - np_label[1::2])]
        offset_sq = offset[0::2] + offset[1::2]
        offset_dist = [math.sqrt(i) for i in offset_sq]
        offset_all.append(offset_dist)
        if test_count == 0:
            drawImageAndCirclesWithPLT(test_image, test_out)
        test_count += 1
        if test_count >= test_limit:
            break
    offset_all = np.asarray(offset_all).flatten()

    itr = 0
    for radius in x_axis:
        y_axis[itr] = sum(i < radius for i in offset_all)
        itr += 1
    x_axis = x_axis / 225
    y_axis = y_axis / len(offset_all)
    plt.plot(x_axis, y_axis, 'ro-')
    plt.xlabel('Radius')
    plt.ylabel('Detected Ratio')
    plt.title('Avg. Percentage of Detected Key-points')
    plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
    plt.show()
