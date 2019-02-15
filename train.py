import numpy as np
import pdb
import re

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import gradcheck
torch.manual_seed(1)

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

from tqdm import tqdm
tqdm.monitor_interval = 0
import logging

import pandas as pd
import csv

import os
import shutil
from datetime import date

import json
from pprint import pprint
import collections
import traceback
###################################
import bidaf_1 as bidaf
import layers
###################################


########################################################################################################################################
'''  Logger  '''
########################################################################################################################################
#Create and configure logger
logging.basicConfig(filename="bidaf.log", format='%(asctime)s %(message)s', filemode='w')
#Creating an object

logger=logging.getLogger()
#Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)
logger.info('Let\'s start logging ')


########################################################################################################################################
'''         LOADING THE DATSETS: SQuAD, Character embeddings, Glove vectors         '''
########################################################################################################################################

### Loading important datasets
### SQUAD dataset
path = os.path.join(os.path.expanduser('~'), 'Bidaf', 'data', 'train-v2.0.json')
with open(path) as f:
    data = json.load(f)

### Character emebeddings
char_model={}
path = os.path.join(os.path.expanduser('~'), 'Bidaf', 'data', 'char-embeddings.txt')
f = open(path, 'r', encoding="utf8")
word=' '
for line in f:
    splitLine = line[1:].split()
    if len(splitLine):
        word = line[0]
        embedding = np.array([float(val) for val in splitLine])
        char_model[word] = embedding

### Loading the glove vectors in the dictionary so that they are easily accessible
## NOTE  that the glove vectors are 50-D dimensional vectors
model={}
path = os.path.join(os.path.expanduser('~'), 'Bidaf', 'data', 'glove.6B.50d.txt')
f = open(path, 'r', encoding="utf8")
for line in f:
    splitLine = line.split()
    word = splitLine[0]
    embedding = np.array([float(val) for val in splitLine[1:]])
    model[word] = embedding

input_size = len(model['try'])
############################################################################################################################################
'''                Getting the SQUAD data-> words to vectors                                   '''
'''
- get_data(): gets you a paragraph , question, answer and answer start
- get_embeddings(): gets you the vector associated with any string
'''
############################################################################################################################################
#Getting the encoded word vectors from
data_count = 0 ## Total 442
paragraph_count = 0
qas_count = 0

def check_data():
        if data_count<=len(data['data']):
                return True
        return False

def check_para():
        if  (paragraph_count<len(data['data'][data_count]['paragraphs'])):
                return True
        return False

def check_ques():
        if qas_count<len(data['data'][data_count]['paragraphs'][paragraph_count]['qas']):
                return True
        return False

def get_data():
        global data_count
        global paragraph_count
        global qas_count

        if check_data():
                if check_para():
                        if check_ques():
                                para, ques, ans1, ans2 = get_para()
                                qas_count+=1

                        else:
                                paragraph_count+=1
                                qas_count = 0
                                if check_para():
                                        para, ques, ans1, ans2 = get_para()
                                        qas_count+=1
                                else:
                                        data_count+=1
                                        paragraph_count=0
                                        qas_count=0
                                        if data_count != 442:
                                                para, ques, ans1, ans2 = get_para()
                                        else:
                                                return -1,-1,-1,-1
                else:
                        data_count+=1
                        paragraph_count=0
                        qas_count=0
                        if data_count!=442:
                                para, ques, ans1, ans2 = get_para()
                        else:
                                return -1,-1,-1,-1
        else:
                return -1,-1,-1,-1

        return para, ques, ans1, ans2



def get_para():
        global data_count
        global paragraph_count
        global qas_count

        para = data['data'][data_count]['paragraphs'][paragraph_count]['context']
        ques = data['data'][data_count]['paragraphs'][paragraph_count]['qas'][qas_count]['question']

        ans= data['data'][data_count]['paragraphs'][paragraph_count]['qas'][qas_count]['answers']
        if ans==[]:
                string = 'plausible_answers'
        else:
                string = 'answers'

        ans_text= data['data'][data_count]['paragraphs'][paragraph_count]['qas'][qas_count][string][0]['text']
        ans_start = data['data'][data_count]['paragraphs'][paragraph_count]['qas'][qas_count][string][0]['answer_start']

        return para, ques, ans_text, ans_start


def character_embeddings(word):
        #Input: a word
        # Find the character embeddings for each character in the word
        #concatenate the character embeddings of all the corresponding characters
        # find the max of across all row and return one vector

        ## These embeddings are then concatenated with the word embeddings

        global char_model
        if word=="":
                return 0.01* np.random.uniform(-3,3,len(char_model['r']))
        if word[0] in char_model:
                x = char_model[word[0]]
        else:
                x =0.01* np.random.uniform(-3,3,len(char_model['r']))

        for i in range(1,len(word)):
                if word[i] in char_model:
                        x =np.column_stack((x, (char_model[word[i]])[:,None]    ))
                else:
                        x =np.column_stack((x,(0.01* np.random.uniform(-3,3,len(char_model['r'])))[:,None]   ))

        if x.ndim!=1:
                return np.max(x, axis=1)
        else:
                return x.reshape(x.shape[0],1)



def get_embeddings(string):
        global model ## accessing GloVe word vectors
        str_1 =  re.sub('[^a-zA-Z0-9\n\.]', ' ',string)
        str_1 = str_1.lower().strip()

        words = str_1.split(' ')
        words =  [i for i in words if i!='']

        matrix =[]
        for i in range(len(words)):
                str_1 ="".join(re.findall("[a-zA-Z0-9]", words[i]))
                if str_1.strip()!="":
                        ##Calling character embeddings for the word -> str_1
                        ch_emd =  character_embeddings(str_1)

                        if str_1 in model:
                                new_d = np.array(model[str_1])
                        else:
                                print('Random accessed for the word: '+ str_1)
                                new_d = 0.01* np.random.uniform(-3,3,len(model['random']))  ### UNK vector for OOV words
                        try:
                                new_d = np.concatenate((new_d.reshape(new_d.shape[0], 1), ch_emd.reshape(ch_emd.shape[0],1)))
                        except Exception:
                                pdb.set_trace()
                        if matrix==[]:
                                matrix= new_d
                        else:
                                try:
                                        matrix = np.column_stack((matrix, new_d))
                                except Exception:
                                        pdb.set_trace()

        if matrix.ndim==1:
                matrix =  matrix.view(matrix.shape[0], 1)
        return matrix, words
        
############################################################################################################################################
'''               CHECK LATER                                  '''
############################################################################################################################################
from sklearn.decomposition import PCA
pca = PCA(n_components=50)

c= 0 ;
letter = []
for i,val in char_model.items():
    letter.append(i)
    if c==0:
        x = val
        c=1
    else:
        x = np.column_stack((x, val[:,None]))

principalComponents = pca.fit_transform(x.T)
### These principal compoenents contain the reduced dimensions for all the letters

chars ={}
for i in range(len(letter)):
    chars[letter[i]] = principalComponents.T[:,i]



############################################################################################################################################
'''                Creating the input batches for training                                 '''
############################################################################################################################################
def pad_the_matrix(t, dim2):
        #padding along the 2nd dimension
        try:
                a = 0.01*np.random.rand(t.shape[0],dim2 - t.shape[1])
        except Exception:
                pdb.set_trace()
        return np.concatenate( (t, a), axis=1 )

###CREATING BATCHES!!!!!!!!!!!!!!!!!!!:
old_para=''
para_emb0=''
words=[]
def get_ans_word_index(words, ans):
      a1 = [i for i in range(len(words)) if words[i:i+len(ans)] == ans]
      return  np.array([a1[0], a1[0]+len(ans)-1 ] )


#Initialization
def get_batches(batch_size, p_dim, q_dim):
        global old_para
        global para_emb0
        global words
        check = 1
        i=1
        while(check):
                para, ques, ans_text, ans_start = get_data()
                if para==-1:
                        return -1,-1,-1, 0
                if para!=old_para:
                        para_emb0,words = get_embeddings(para)
                        old_para = para
                ques_emb0, _  = get_embeddings(ques)
                ans_emb0, ans = get_embeddings(ans_text)
                
                if (p_dim>=para_emb0.shape[1]) & (q_dim>=ques_emb0.shape[1]):

                        ####Padding occurs here:
                        para_emb = pad_the_matrix(para_emb0, p_dim)
                        ques_emb = pad_the_matrix(ques_emb0, q_dim)

                        X = torch.Tensor(para_emb.reshape(para_emb.shape[1], 1, para_emb.shape[0]))
                        Q = torch.Tensor(ques_emb.reshape(ques_emb.shape[1], 1, para_emb.shape[0]))
                        ans_start_end = get_ans_word_index(words, ans)
                        check=0
        while(i<batch_size):
                para, ques, ans_text, ans_start = get_data()
                if para==-1:
                        return X,Q, ans_start_end, 0

                para_emb0, words = get_embeddings(para)
                ques_emb0, _  = get_embeddings(ques)
            ###Need to pad the data because the length of the context and the questions might difffer
            ## Basic anlaysis : considerign 1000 as the length for the context and
            ## 100 as thje length for the question
            ## If the size of the question or the context is greater than the above decided parameters,
            ## Then basically icbs(target[i]- predicted[i].type(torch.FloatTensor))
            #core the question and answer combo ---> moving on.. modify latyer
                if (p_dim>=para_emb0.shape[1]) & (q_dim>=ques_emb0.shape[1]):
                        ####Padding occurs here:
                        para_emb = pad_the_matrix(para_emb0, p_dim)
                        ques_emb = pad_the_matrix(ques_emb0, q_dim)

                        ans_emb0, ans = get_embeddings(ans_text)

                        a1= torch.Tensor(para_emb.reshape(para_emb.shape[1], 1, para_emb.shape[0]))
                        a2 = torch.Tensor(ques_emb.reshape(ques_emb.shape[1], 1, para_emb.shape[0]))

                        X = np.concatenate((X, a1), axis=1)
                        Q = np.concatenate((Q, a2), axis=1)
                        ans_start_end = np.vstack((ans_start_end, get_ans_word_index(words, ans)))
                        i+=1
        return X, Q, ans_start_end,1

############################################################################################################################################
'''             Finally the treaining begins here!                                 '''
############################################################################################################################################
''' Creating a checkpoint  file'''


def save_checkpoint(state, filename='checkpoint.pth.tar'):
	string =  os.path.join('Bidaf/epoch_models/', filename)
	torch.save(state, filename)

#Hyperparameters
epochs=200
epoch_break = 1

p_dim =125 
q_dim = 13


input_size = 350
hidden_size =25 
num_layers =2
batch_size =5
bidirectional =2

#criterion = torch.nn.MarginRankingLoss(margin = args.margin)
bidaf_model = bidaf.model(input_size, hidden_size,num_layers, batch_size, bidirectional)
bidaf_model.cuda()

if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        bidaf_model = nn.DataParallel(bidaf_model)

#optimizer = optim.SGD(bidaf_model.parameters(), lr = 0.2, momentum=0.9)
optimizer = torch.optim.Adam(bidaf_model.parameters(), lr=0.01)
optimizer.zero_grad()
loss_list = []
print('Training begins')
for ee in tqdm(range(epochs)):
	print('##################################################### \nEpoch: ' +  str(ee) + '\n')
	#log file:  trianing time, epoch number, loss calculated
	for iters in range(int(90000/batch_size)):
		loss=0
		optimizer.zero_grad()
		bidaf_model.zero_grad()
		X,Q,target1, epoch_break = get_batches(batch_size, p_dim, q_dim)
		X=torch.Tensor(X).cuda()
		Q=torch.Tensor(Q).cuda()

		p1_, p2_ = bidaf_model(X,Q)
		target=torch.Tensor(target1)

		for ba in tqdm(range(target.shape[0])):
		       loss+=-torch.log( p1_[int(target[ba,0]) ,ba]   *  p2_[int(target[ba,1]),ba] )

		loss_string = 'loss at epoch '+ str(ee) +  'and iteration '+ str(iters)  +' is: '+ str(loss)
		print('\n\n\n\n\n\n')
		print(loss_string)
		logger.info(loss_string)
		logger.info('/n')
		
		torch.nn.utils.clip_grad_norm_(bidaf_model.parameters(),1)
		loss.sum().backward()
		#print(gradcheck(bidaf_model, (p1_)))
		optimizer.step()
        
	print('SAVED!!!!!')
	save_checkpoint({'epoch': ee + 1,'state': bidaf_model.state_dict(), 'loss': loss_list, 'optimizer' : optimizer.state_dict(),}, '3_model_at_epoch_'+str(ee))

	data_count = 0 ## Total 442
	paragraph_count = 0
	qas_count = 0

