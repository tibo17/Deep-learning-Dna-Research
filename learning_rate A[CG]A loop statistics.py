import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.metrics import accuracy_score
from tqdm import tqdm

accuracy_list = []


## parameters of the neural network
num_epochs = 200
batch_size_train = 5

#the batch size of train set correspond to the number of sample in train set
batch_size_test = 100



pattern = "A[CG]A"
size_pattern = 3

length_sequence = 25


nb_test_samples = 100
nb_train_samples = 1000


##Dna class used to create the dataset
class Dna(Dataset):

    def __init__(self, pattern, length_sequence, nb_samples):

        self.input = dataset_input_creation(pattern, length_sequence, nb_samples)
        self.output = dataset_output_creation(nb_samples)
        self.pattern = pattern
        self.length_sequence = length_sequence
        self.nb_samples = nb_samples


    def __getitem__(self, index):
        return self.input[index], self.output[index]


    def __len__(self):
        return self.nb_samples

##ConvNet neural network class

class ConvNet(nn.Module):

    def __init__(self,size_pattern):
        self.size_pattern = size_pattern
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(4, 1, size_pattern)
        self.conv2 = nn.Conv1d(1, 1, size_pattern)
        self.fc = nn.Linear(length_sequence-2*(size_pattern-1), 1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, length_sequence-2*(self.size_pattern-1))
        x = F.relu(self.fc(x))
        return x


##begining of the loop

lr_list = np.arange(0.001, 0.03, (0.03 - 0.001)/ 50)

accuracy_lr_list = []
for learning_rate in tqdm(lr_list):


    accuracy_list = []


    for k in range (10):




        ##Creation of the dataset

        #data for the training loop

        dataset = Dna(pattern, length_sequence, nb_train_samples)

        dataloader = DataLoader(dataset = dataset, batch_size = batch_size_train, shuffle = True)

        '''dataiter = iter(dataloader)

        input, labels = dataiter.next()'''

        total_samples = len(dataset)

        #data for the test
        datasettest =  Dna(pattern, length_sequence, nb_test_samples)

        dataloadertest = DataLoader(dataset = datasettest, batch_size = batch_size_test, shuffle = True)




        ## Training loop

        model = ConvNet(size_pattern)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)




        for epoch in range(num_epochs):
            for i, (input_train, label_train) in enumerate(dataloader):


                prediction_train = model(input_train)
                loss = nn.MSELoss()
                l = loss(label_train, prediction_train)

                optimizer.zero_grad()
                l.backward()
                optimizer.step()

        ## Test loop

        for epoch in range(1):
            #batch_size = nb_test_samples, so the loop run just one time.
            for i, (input_test, label_test) in enumerate(dataloadertest):


                prediction_test = model(input_test)
                loss = nn.MSELoss()
                l = loss(label_test, prediction_test)


                #transformation of the predictions tensor with float in predition tensor with 0 and ones
                for i in range(prediction_test.size(0)):
                    with torch.no_grad():

                        if prediction_test[i] > 0.5:
                            prediction_test[i] = torch.tensor([1.])
                        else:
                            prediction_test[i]= torch.tensor([0.])

                #beacause there is just one calculation of accuracy (loop run one time beacause nb_test_samples = batc_size_test), accuray test can be calculate outside of the loop.
                with torch.no_grad():

                    accuracy_test = accuracy_score(prediction_test, label_test)
        print(accuracy_test)

        accuracy_list.append(accuracy_test)
        print(k)

    accuracy_list = np.array(accuracy_list)

    accuracy_mean = accuracy_list.mean()
    accuracy_std = accuracy_list.std()
    accuracy_median = np.median(accuracy_list)
    accuracy_min = min(accuracy_list)
    accuracy_max = max(accuracy_list)



    print(f'accuracy_mean: {accuracy_mean}')
    print(f'accuracy_list: {accuracy_list}')
    print(f'accuracy_std: {accuracy_std}')
    print(f'accuracy_median: {accuracy_median}')
    print(f'accuracy_min: {accuracy_min}')
    print(f'accuracy_max: {accuracy_max}')

    accuracy_lr_list.append(accuracy_mean)


plt.figure()
plt.plot(lr_list, accuracy_lr_list)
plt.title("accuracy in relation to the learning rate")
plt.xlabel("learning rate")
plt.ylabel("accuracy")
plt.show()

