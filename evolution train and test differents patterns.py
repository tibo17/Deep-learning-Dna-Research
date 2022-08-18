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
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import exrex





moyenne_accuracy_list = []
list_filters = []


## parameters of the neural network
nb_epochs = 200
batch_size_train = 5

#the batch size of train set correspond to the number of sample in train set
batch_size_test = 100

learning_rate = 0.01

pattern_list =  ["TTCT", "CACGTTGCCGGGGGCGAGT", "C[ACT][ACGT][ACGT]A[GCA][TA]", "TGT[ACGT]GGT[TG]AGGCG"]
size_pattern = 3
size_pattern_list = [len(exrex.getone(pattern)) for pattern in pattern_list]
length_sequence = 40


nb_test_samples = 100
nb_train_samples = 1000





## parameters accuracy


accuracy_evolution_train = []
accuracy_evolution_test = []

'''accuracy_list = [[i] for i in pattern_list]'''



accuracy_list = []



list_epochs = [i for i in range(nb_epochs)]





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

    def __init__(self, size_pattern):
        self.size_pattern = size_pattern
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(4, 1 , size_pattern)
        self.conv2 = nn.Conv1d(1, 2, size_pattern)
        self.fc = nn.Linear(2*(length_sequence-2*(size_pattern-1)), 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 2*(length_sequence-2*(self.size_pattern-1)))
        x = F.relu(self.fc(x))
        return x


## training loop



'''loss = nn.CrossEntropyLoss()'''
plt.figure()


'''nb_true_prediction = 0'''


for pattern in pattern_list:

    ##Creation of the dataset



    #data for the training loop
    dataset = Dna(pattern, length_sequence, nb_train_samples)

    dataloader = DataLoader(dataset = dataset, batch_size = batch_size_train, shuffle = True)

    #data for the test
    dataset_test =  Dna(pattern, length_sequence, nb_test_samples)

    dataloader_test =  DataLoader(dataset = dataset_test, batch_size = batch_size_test, shuffle = True)

    ##Initialisation of the neural network

    model = ConvNet(size_pattern_list[pattern_list.index(pattern)])
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    print(' ')
    print(f"begining of training for pattern {pattern}")
    print(' ')



    for epoch in tqdm(range(nb_epochs)):


        accuracy_evolution_train_epoch = []

        accuracy_evolution_test_epoch = []



        for nb_training , (input_train, label_train) in enumerate(dataloader):



            #calculation of prediction of neural network and loss for train data
            prediction_train = model(input_train)
            loss = nn.MSELoss()
            l = loss(label_train, prediction_train)



            #updating of the network

            optimizer.zero_grad()
            l.backward()
            optimizer.step()


            #calculation of the accuracy for test data and train data
            if nb_training%2 == 0:


                #calculation of the accuracy for test data
                #batch size is equal to nb test sample, so, this loop run one time

                for i, (input_test,label_test) in enumerate(dataloader_test):


                    prediction_test = model(input_test)

                    with torch.no_grad():

                        #transformation of the predictions tensor with float in predition tensor with 0 and ones

                        for i in range(prediction_test.size(0)):

                            if prediction_test[i] > 0.5:
                                prediction_test[i] = torch.tensor([1.])
                            else:
                                prediction_test[i]= torch.tensor([0.])

                        #calcul of accuracy for each prediction, and adding to a list wich contain all test accuracies scores for this epoch

                        accuracy_test = accuracy_score(prediction_test, label_test)


                        accuracy_evolution_test_epoch.append(accuracy_test)


                #calculation of the accuracy for train data
                with torch.no_grad():
                    #transformation of the predictions tensor with float in predition tensor with 0 and ones

                    for i in range(prediction_train.size(0)):

                        if prediction_train[i] > 0.5:
                            prediction_train[i] = torch.tensor([1.])
                        else:
                            prediction_train[i]= torch.tensor([0.])




                    #calcul of accuracy for each prediction, and adding to a list wich contain all train accuracies scores for this epoch



                    accuracy_training = accuracy_score(prediction_train, label_train)



                    accuracy_evolution_train_epoch.append(accuracy_training)










        #the accuracy for one epoch correspond to the mean of the list accuracy_evolution_epoch (for train and test)



        accuracy_training = sum(accuracy_evolution_train_epoch) / len(accuracy_evolution_train_epoch)

        accuracy_test = sum(accuracy_evolution_test_epoch) / len(accuracy_evolution_test_epoch)


        accuracy_evolution_train.append(accuracy_training)

        accuracy_evolution_test.append(accuracy_test)



    plt.plot(list_epochs, accuracy_evolution_train, label = f'train {pattern}')
    plt.plot(list_epochs, accuracy_evolution_test, label = f'test {pattern}')

    accuracy_evolution_train = []
    accuracy_evolution_test = []


#creation of the plot representing the evolution of accuracy through training
plt.rc('font', size= 6)
plt.title(f'accuracy evolution for train set and test set')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.show()















