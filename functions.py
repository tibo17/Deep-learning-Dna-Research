import random
import string


import exrex

import re
import torch
import numpy as np

from tqdm import tqdm


def sequence_creation_without_pattern(length_sequence, pattern):



    '''Creation of the sequence string'''
    sequence = ''.join(random.choice("ACGT") for i in range(length_sequence))

    

    '''We initialise pattern already here to True'''
    pattern_already_here = True

    '''This loop stop only when the sequence have no pattern possibility in it'''
    while(pattern_already_here):

        pattern_already_here = re.search(pattern, sequence)

        if pattern_already_here:
            sequence = ''.join(random.choice("ACGT") for i in range(length_sequence))


    return sequence




def sequence_creation_with_pattern(length_sequence, pattern):

    '''Creation of a sequence without pa(ttern by calling the function sequence_creation_without_pattern'''
    sequence = sequence_creation_without_pattern(length_sequence, pattern)

    ''' random determination of a fix pattern'''

    pattern = exrex.getone(pattern)

    ''' determination of a podition for the pattern in the sequence'''
    length_pattern = len(pattern)
    position_pattern = random.randint(0, length_sequence - length_pattern)

    ''' adding of the pattern in the sequence'''
    sequence = sequence[:position_pattern] + pattern + sequence[position_pattern + length_pattern:]


    return sequence





def matrix_creation(sequence):
    ''' this function create a matrix which correspond to the sequence'''
    length_sequence = len(sequence)
    matrix = np.zeros((4, length_sequence))
    for position_lettre in range(length_sequence):

        lettre = sequence[position_lettre]


        if (lettre == "A"):

            matrix[0,position_lettre] = 1

        if (lettre == "C"):

            matrix[1,position_lettre] = 1

        if (lettre == "G"):

            matrix[2,position_lettre] = 1

        if (lettre == "T"):

            matrix[3,position_lettre] = 1


        matrixtensor = torch.Tensor(matrix)

    return matrixtensor







def dataset_input_creation(pattern, length_sequence, nb_samples):

    ''' this fucntion create a input dataset with half of the data without the pattern, and half of the data with   it'''
    data_input_list =  []


    for loop in range (nb_samples//2):

        sequence = sequence_creation_without_pattern(length_sequence, pattern)
        matrix = matrix_creation(sequence)

        data_input_list.append(matrix)



    for loop in range(nb_samples//2):

        sequence = sequence_creation_with_pattern(length_sequence, pattern)
        matrix = matrix_creation(sequence)
        data_input_list.append(matrix)


    return data_input_list



def dataset_output_creation(nb_samples):

    ''' this fucntion create a output dataset. It is a one column tesnor where half parameters are zeros, half are one '''
    data_output_list =  []

    for loop in range(nb_samples//2):
         data_output_list.append([0.])

    for loop in range(nb_samples//2):

        data_output_list.append([1.])

    data_output = torch.Tensor(data_output_list)

    return data_output






def ratio_pattern_noise(pattern, length_sequence):


    pattern = exrex.getone(pattern)

    return len(pattern) / length_sequence

def probability_of_noise(pattern):

    '''number of possibility of pattern'''
    nb_possibility_pattern = exrex.count(pattern)
    pattern = exrex.getone(pattern)
    length_pattern = len(pattern)


    '''number of possibility for a sequence with same length as the pattern'''
    nb_possibility_sequence = 4**length_pattern

    return nb_possibility_pattern/ nb_possibility_sequence




def crea_pattern_proba(proba,length_pattern):

    '''this function create a random pattern with a chosen law of probability and a chosen length'''
    '''the variable proba is a list of four elements. Ech element correspond to a probability for a certain type of character: simple letter, two letters in brackets, three letters in bracket, four letters in bracket'''

    from random import random


    four_letters_bracket_list =["[ACGT]"]

    three_letters_bracket_list =["[ACT]", "[GCA]", "[TAG]", "[GTC]"]

    two_letters_bracket_list = ["[AG]", "[TA]", "[AC]", "[GC]", "[TG]", "[CT]"]

    letters_list = ["A", "C", "G", "T"]

    possibilities = [letters_list, two_letters_bracket_list, three_letters_bracket_list, four_letters_bracket_list]

    num_list = [1,2,3,4]

    assert round(np.sum(proba), 3) == 1.0
    assert len(possibilities) == len(proba)

    bad_pattern = True

    while bad_pattern == True:

        '''bad pattern is true when we have "[ACGT]" in first or last position'''
        pattern = ""
        '''this loop create the pattern with the rigth law of probability'''
        for loop in range(length_pattern):
            from random import random
            nb=random()
            curseur=0
            for i in range(len(proba)):
                if curseur<=nb<curseur+proba[i]:
                    import random
                    new_character = random.choice(possibilities[i])
                    pattern = pattern + new_character
                    break
                curseur=curseur+proba[i]
        '''this loop verify if there [ACGT] in first or last podition'''

        try:

            begining_characters = ""
            for i in range(6):
                begining_characters += pattern[i]

            ending_characters = ""
            for i in range(6):
                ending_characters += pattern[len(pattern)- 6 +i]

            if ((begining_characters != "[ACGT]" ) and (ending_characters != "[ACGT]")):
                bad_pattern = False

        except IndexError:
            '''the index error is here when the length pattern is < 6, consequently, the pattern can't have [ACGT] in the last position inn this case'''

            return pattern


    return pattern



def crea_pattern_p_noise(p_noise, length_pattern):

    '''this function create a random pattern of a p_noise and a length of our choice'''

    '''lits_proba is a list of differents laws of probability, with more and more probability of noise'''

    list_proba =np.array([[9-i, 0.8+ i, 0.4+i**2, 0.2+i**3] for i in np.linspace(0, 8, 50)])

    list_proba = [list_proba[i]/sum(list_proba[i]) for i in range(len(list_proba))]

    pattern_creation_list = []


    p_noise_pattern = 1.0e-25

    if p_noise_pattern > p_noise:

        print("the value of p_noise is too low")

    num_proba = 0

    '''This loop run until a pattern which exceed the p_noise given in parameters is found'''

    while (len(pattern_creation_list) == 0):

        pattern_creation_list = []

        '''This loop create 100 pattern, and create a list of all patterns which exceed the p_noise choose in parameters'''

        '''If no patterns which exceed the p_noise are found in the 100 samples, we increase num_proba in order to have a proba of p_noise higher, and the loop is run again'''

        for loop in range(100):

            pattern = crea_pattern_proba(list_proba[num_proba], length_pattern)


            p_noise_pattern = probability_of_noise(pattern)

            if (p_noise_pattern >= p_noise):

                pattern_creation_list.append(pattern)

        num_proba += 1

    '''Next, when pattern are found, we choose the pattern with the minimum p_noise of the list, to have a p_noise as close as possible to the p_noise chosen in parameters'''


    pattern_pnoise_creation_list = [probability_of_noise(pattern) for pattern in pattern_creation_list]

    p_noise_pattern = min(pattern_pnoise_creation_list)

    index_pattern = pattern_pnoise_creation_list.index(p_noise_pattern)

    pattern = pattern_creation_list[index_pattern]


    return pattern








