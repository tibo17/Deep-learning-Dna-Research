# point comet_ml SDK to www.comet.com installation
import os
os.environ["COMET_URL_OVERRIDE"] = "https://www.comet.com/clientlib/"



# Import comet_ml at the top of your file
from comet_ml import Experiment

# Create an experiment with your api key
experiment = Experiment(
    api_key="dprRh9BVcscKSYxOB92K9q7vR",
    project_name="general"
)

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
from functions import*


moyenne_accuracy_list = []
list_filters = []



    ## parameters of the neural network
nb_epochs = 200


#the batch size of train set correspond to the number of sample in train set

#the hyper params are fixed and will never change
hyper_params = {
    "batch_size_train":10,
    "batch_size_test": 100,
    "learning_rate":0.01,

    "length_sequence":40,

    "nb_train_samples": 400,

    "nb_test_samples": 100,

    "nb_pattern_per_size": 15,
    "size_pattern_max":  20,
    "size_pattern_min":  4
    }


batch_size_train = hyper_params["batch_size_train"]
batch_size_test = hyper_params["batch_size_test"]
learning_rate = hyper_params["learning_rate"]

length_sequence = hyper_params["length_sequence"]


experiment.log_parameters(hyper_params)

nb_train_samples = hyper_params["nb_train_samples"]
nb_test_samples = hyper_params["nb_test_samples"]

nb_pattern_per_size = hyper_params["nb_pattern_per_size"]
size_pattern_max = hyper_params["size_pattern_max"]
size_pattern_min = hyper_params["size_pattern_min"]



pattern_list = []
p_noise_list = []
rpn_list = []
size_pattern_list = []



for size_pattern in range(size_pattern_min, size_pattern_max+1):


    #min_p_noise is the minimum probability of noise possible for a determined size pattern
    min_p_noise = probability_of_noise(''.join(random.choice("ACGT") for i in range(size_pattern)))

    #the script which follow creat a list of p_noise sorted in ascending order, for a determined size pattern, in order to have all p_noise possible, from the smalest to the highest

    coeff = (0.01 / min_p_noise)**(1/(nb_pattern_per_size-1))

    list_p_noise = [min_p_noise]
    p_noise = min_p_noise

    for loop in range(nb_pattern_per_size-1):
        p_noise = p_noise* coeff
        list_p_noise.append(p_noise)


    '''this loop create the pattern_list'''

    for i in range(nb_pattern_per_size):

        p_noise = list_p_noise[i]
        a = crea_pattern_p_noise(p_noise, size_pattern)
        pattern_list.append(a)
        size_pattern_list.append(size_pattern)




#the index of each elements in p_noise_list and rpn_list correspond to the index of the pattern associated in pattern_list

p_noise_list = [probability_of_noise(i) for i in pattern_list]
rpn_list = [ratio_pattern_noise(i, length_sequence) for i in pattern_list]

print(" ")
print(pattern_list)
print(" ")
print(p_noise_list)
print(" ")
print(rpn_list)
print(" ")
print(size_pattern_list)




experiment.log_parameter("pattern_list", pattern_list)
experiment.log_parameter("p_noise_list", p_noise_list)
experiment.log_parameter("rpn_list", rpn_list)


nb_pattern = len(pattern_list)

experiment.log_parameter("nb_pattern", nb_pattern)


#We run the following loop nb_try times, consequently, the accuracy find for each pattern is  the mean accuracy of all of the nb_try iterations

nb_try = 15
experiment.log_parameter("nb_try", nb_try)
matrix_accuracy = np.zeros((nb_pattern, nb_try))


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
        self.conv2 = nn.Conv1d(1, 1, size_pattern)
        self.fc = nn.Linear(length_sequence-2*(size_pattern-1), 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, length_sequence-2*(self.size_pattern-1))
        x = F.relu(self.fc(x))
        return x



##Lopp which calculate the mean accuracy for each pattern in  order to create the graph

for num_try in tqdm(range(nb_try)):

    ## parameters accuracy

    experiment.log_metric("num_try", num_try, step=num_try)


    accuracy_evolution_train = []
    accuracy_evolution_test = []

    '''accuracy_list = [[i] for i in pattern_list]'''



    accuracy_list = []
    accuracy_std_list = []
    accuracy_median_list = []
    accuracy_min_list = []
    accuracy_max_list = []



    list_epochs = [i for i in range(nb_epochs)]

    

    ## training and testing loop, calculation of accuracy for each pattern




    for pattern in tqdm(pattern_list):

        nb_true_prediction = 0


        #creation of data

        print("creation of data for probability of noise:", probability_of_noise(pattern), "and ration pattern noise", ratio_pattern_noise(pattern, length_sequence))

        #data for the training loop
        
        dataset = Dna(pattern, length_sequence, nb_train_samples)
        
        dataloader = DataLoader(dataset = dataset, batch_size = batch_size_train, shuffle = True)
        
        #data for the test
        dataset_test =  Dna(pattern, length_sequence, nb_test_samples)

        dataloader_test =  DataLoader(dataset = dataset_test, batch_size = batch_size_test, shuffle = True)
        
        #initialisation of neural network
        model = ConvNet(size_pattern_list[pattern_list.index(pattern)])
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        

        print("begining of the training for probability of noise", probability_of_noise(pattern), "and ration pattern noise", ratio_pattern_noise(pattern, length_sequence))


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



        '''plt.plot(list_epochs, accuracy_evolution_train, label = f'train {pattern}')
        plt.plot(list_epochs, accuracy_evolution_test, label = f'test {pattern}')'''









        #this loop calculate the accuracy for the test data to create the graphs representing the accuracy in relation to the probability of noise and ratio pattern noise.
        for epoch in range(1):

            #batch_size = nb_test_samples, so the loop run just one time.
            for i, (input_test, label_test) in enumerate(dataloader_test):

                prediction_test = model(input_test)


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

                

                    

                #the accuracy is added to accuracy_list. The index correspond to the pattern associated in pattern list.
                accuracy_list.append(accuracy_test)

                #the accuracy is added to matrix_accuracy. The first index correspond to the pattern associated in pattern list. The second index correspond to the number of try corresponding.

                matrix_accuracy[pattern_list.index(pattern)][num_try] = accuracy_test


        accuracy_evolution_train = []

        accuracy_evolution_test = []


    ''' #creation of the plot representing the evolution of accuracy through training
    plt.rc('font', size= 6)
    plt.title(f'accuracy evolution for differents pattern')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    experiment.log_figure(figure=plt)
    plt.show()'''




    '''moyenne_accuracy_list.append(sum(accuracy_list) / len(accuracy_list))'''

#we transform all matrix accuracy lines into their mean.
for index_pattern in range(len(matrix_accuracy)):

    accuracy_list.append(matrix_accuracy[index_pattern].mean())
    accuracy_std_list.append(matrix_accuracy[index_pattern].std())
    accuracy_median_list.append(np.median(matrix_accuracy[index_pattern]))
    accuracy_min_list.append(min(matrix_accuracy[index_pattern]))
    accuracy_max_list.append(max(matrix_accuracy[index_pattern]))



        




#creation of lists. Each list correspond to the rpn and the pnoise for one interval of accuracy.

print(' ')
print(f'accuracy_std_list: {accuracy_std_list}')
print(' ')
print(f'pattern_list: {pattern_list}')
print(' ')
print(f'matrix_accuracy: {matrix_accuracy}')
print(' ')
print(f'p_noise_list: {p_noise_list}')
print(' ')
print(f'rpn_list: {rpn_list}')
print(' ')
print(f'accuracy_list: {accuracy_list}')
print(' ')
print(f'accuracy_median_list: {accuracy_median_list}')
print(' ')
print(f'accuracy_min_list: {accuracy_min_list}')
print(' ')
print(f'accuracy_max_list: {accuracy_max_list}')
print(' ')

print('predictionzerolist: {predictionzerolist}')
print(' ')
print('labelzerolist: {labelzerolist}')
print(' ')



print('creation of the scatter plot')




##creation of plots

#Creation of the plot representing the mean accuracy and standard deviation of accuracy in 15 trainings in relation to the probability of noise and ratio pattern noise


fig, ax = plt.subplots()


sc = plt.scatter(rpn_list, p_noise_list, c= accuracy_list, cmap='RdYlBu', s = (np.array(accuracy_std_list)*100)**2)

plt.title(f'mean accuracy and standard deviation of accuracy in {nb_try} trainings in relation to the probability of noise and ratio pattern noise')
kw = dict(prop ="sizes", num = 8, color = "grey", fmt =" {x:.2f}", func = lambda s: np.sqrt(s)/100)
legend = ax.legend(*sc.legend_elements(**kw), loc ="lower left", title ="Standard deviation")
ax.add_artist(legend)



plt.yscale("log")
plt.xlabel('ratio pattern noise')
plt.ylabel('probability of noise')
colorbar = plt.colorbar()
colorbar.set_label('accuracy')
experiment.log_figure(figure=plt)

#Creation of the plot representing the max accuracy in 15 training in relation to the probability of noise and ratio pattern noise

plt.figure()
plt.title(f'max accuracy in {nb_try} training in relation to the probability of noise and ratio pattern noise')
plt.scatter(rpn_list, p_noise_list, c= accuracy_max_list, cmap='RdYlBu', s = 20)
plt.yscale("log")
plt.xlabel('ratio pattern noise')
plt.ylabel('probability of noise')
colorbar = plt.colorbar()
colorbar.set_label('max accuracy in 15 trainings')
experiment.log_figure(figure=plt)


#Creation of the plot representing the min accuracy in 15 training in relation to the probability of noise and ratio pattern noise

plt.figure()
plt.figure()
plt.title(f'min accuracy in {nb_try} training in relation to the probability of noise and ratio pattern noise')
plt.scatter(rpn_list, p_noise_list, c= accuracy_min_list, cmap='RdYlBu', s = 20)
plt.yscale("log")
plt.xlabel('ratio pattern noise')
plt.ylabel('probability of noise')
colorbar = plt.colorbar()
colorbar.set_label('min accuracy in 15 trainings')


experiment.log_figure(figure=plt)
plt.show()


