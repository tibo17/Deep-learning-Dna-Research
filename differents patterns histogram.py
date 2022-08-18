import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import math
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import exrex




pattern_list = ["AAAA", "TAA[ACT]T", "AATTA[TA]", "[AC][ACGT]G[TA][CT][CT]C", "ACT[GTC]C[GTC]C[GTC]", "C[TAG][ACT][ACT][ACGT][ACGT][GCA][ACGT]G"]

size_pattern_list = [len(exrex.getone(pattern)) for pattern in pattern_list]

accuracy_mean_list = []
accuracy_std_list = []

for pattern in tqdm(pattern_list):

    print('')
    print(f'begining of the training for pattern {pattern}')
    print('')


    accuracy_list = []

    size_pattern = size_pattern_list[pattern_list.index(pattern)]


    for k in range (10):
        print('')
        print(f'begining of loop {k+1}')
        print('')


        ## parameters of the neural network
        num_epochs = 200
        batch_size_train = 5

        #the batch size of train set correspond to the number of sample in train set
        batch_size_test = 100

        learning_rate = 0.01



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

            def __init__(self, size_pattern):
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


        ##Creation of the dataset


        #data for the training loop

        dataset = Dna(pattern, length_sequence, nb_train_samples)

        dataloader = DataLoader(dataset = dataset, batch_size = batch_size_train, shuffle = True)


        total_samples = len(dataset)

        #data for the test loop
        datasettest =  Dna(pattern, length_sequence, nb_test_samples)

        dataloadertest = DataLoader(dataset = datasettest, batch_size = batch_size_test, shuffle = True)

        ## Training loop

        model = ConvNet(size_pattern)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)




        for epoch in tqdm(range(num_epochs)):
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
                '''print(" ")
                print(prediction_test)
                print(label_test)
                print(l)
                print(" ")'''


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


        accuracy_list.append(accuracy_test)


        print('')
        print(f'end of loop {k+1}')
        print('')

    accuracy_list = np.array(accuracy_list)

    accuracy_mean = accuracy_list.mean()
    accuracy_std = accuracy_list.std()
    accuracy_median = np.median(accuracy_list)
    accuracy_min = min(accuracy_list)
    accuracy_max = max(accuracy_list)

    print('')
    print(f'end of the training for pattern {pattern}')
    print('')
    print(f'statisitcs for {pattern}')
    print('')

    print(f'accuracy_mean: {accuracy_mean}')
    print(f'accuracy_list: {accuracy_list}')
    print(f'accuracy_std: {accuracy_std}')
    print(f'accuracy_median: {accuracy_median}')
    print(f'accuracy_min: {accuracy_min}')
    print(f'accuracy_max: {accuracy_max}')

    accuracy_list.tolist()

    accuracy_mean_list.append(accuracy_mean)
    accuracy_std_list.append(accuracy_std)



##Creation of the plot representing the histogram of the test data accuracies for the differents patterns

#Creation of the list of colors. Each color represent an interval of standart deviation. Each index of the list  of colors correspond to the index of standard deviation associated in accuracy_std_list.

color_list = []

for i in accuracy_std_list:

    if i < 0.04:
        color_list.append('red')

    elif i < 0.07:

        color_list.append('orange')

    elif i < 0.1:

        color_list.append('yellow')

    else:
        color_list.append('green')


red_patch = mpatches.Patch(color='red', label='standard deviation: 0%, 4%')
orange_patch = mpatches.Patch(color='orange', label='standard deviation: 4%, 7%')
yellow_patch = mpatches.Patch(color='yellow', label='standard deviation: 7%, 10%')
green_patch = mpatches.Patch(color='green', label='standard deviation > 10%')






plt.rc('font', size= 6)
'''fig, axes = plt.subplots(figsize=(30, 30), dpi=100)'''
plt.bar(pattern_list, accuracy_mean_list, color = color_list)

plt.xlabel("pattern")
plt.ylabel("accuracy")

plt.title(f'Mean accuracies and standard deviation in 10 trainings for differents patterns')
plt.legend(handles=[red_patch, orange_patch, yellow_patch, green_patch])


plt.show()



