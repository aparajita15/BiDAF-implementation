'''
BIDAF Model
'''
import numpy as np
import pdb
import traceback
import re
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
torch.manual_seed(1)
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import pandas as pd
import csv
import os
from datetime import date
import json
from pprint import pprint
import collections

from tqdm import tqdm

class model(nn.Module):
	def __init__(self, input_size, hidden_size,num_layers, batch_size, bidirectional):
		super(model, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.batch_size = batch_size
				
		self.bilstm_query = nn.LSTM(input_size, hidden_size,num_layers,bias=True,dropout=0.2,  bidirectional=True)
		self.bilstm_context =  nn.LSTM(input_size, hidden_size,num_layers,bias=True,dropout=0.2,  bidirectional=True)
		
		self.bilstm_query.zero_grad()
		self.bilstm_context.zero_grad()
		d = hidden_size
		self.att_model = nn.Linear(6*d, 1, bias=True)
		self.att_model1 = nn.Linear(8*d, 8*d, bias= True)

		self.bi_lstm2 =  nn.LSTM(8*d, hidden_size,num_layers,bias=True,dropout=0.2,  bidirectional=True)
		self.bi_lstm3 = nn.LSTM(2*d, hidden_size,num_layers,bias=True,dropout=0.2,  bidirectional=True)

		self.out_layer_1 =nn.Linear(10*d, 1,bias = True)
		self.out_layer_2 = nn.Linear(10*d, 1, bias = True)

		self.dropout=nn.Dropout(0.2)

	def forward(self, X, Q):
		T = X.shape[0]
		J = Q.shape[0]
		d= self.hidden_size
		self.bilstm_query.flatten_parameters()
		self.bilstm_context.flatten_parameters()
		self.bi_lstm2.flatten_parameters()
		self.bi_lstm3.flatten_parameters()


		H,_ = self.bilstm_context(X, None) # T,B,2* hidden_size
		U,_ =self.bilstm_query(Q,None)   # J, B, 2*hidden_size
	
	
		B = H.shape[1]
		S = torch.zeros(T,B,J, requires_grad=True).cuda()
		for i in range(T):
			for j in range(J):
				for b in range(B):
					input_data = torch.cat([H[i,b,:], U[j,b,:], H[i,b,:]*U[j,b,:]])
					S[i,b,j] = self.att_model(input_data)
		#pdb.set_trace()
		######### Attention layers - context2query layer
		## a<t>  : J dimesional
		## a<t>  =softmax(S<t,:>)
		## Query vector U_tilda = Sum_over_j( a<t,j> , U<:,j> )   : 2d x T
		#print('Attention layers- context2Query')
		
		a = torch.zeros(S.shape,requires_grad=True).cuda()
		for t in range(T):
			for b in range(B):
				a[t,b,:] = nn.functional.softmax(S[t,b,:], dim=0 )
		U_tilda = torch.zeros((2*d,B, T)).cuda()
		for t in range(T):
			for j in range(J):
				for ba in range(B):
					try:
						U_tilda[:,ba,t] += a[t,ba,j]*U[j,ba,:]
					except Exception:
						print('CHECK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
						pdb.set_trace()


		######### Attention layers - query2context layer
		## b  : T dimesional
		## b  =softmax(max_col((S<t,:>)))
		## vector h_tilda = Sum_over_t( b , H<t,:> )   : 2d dimensions
		## H_tilda = h_tilda repeated T times to give a matrix of dimensionality 2d x T
		#print('Attention layer - Query2Context')
		#pdb.set_trace()
		b= torch.zeros((T, B), requires_grad=True).cuda()
		for ba in range(B):
			b[:, ba], _ =S[:,ba,:].max(1)


		h_tilda = torch.zeros((2*d, B), requires_grad=True).cuda()
		for t in range(T):
			for ba in range(B):
				h_tilda[:, ba] += b[t,ba]*H[t,ba,:]


		H_tilda = torch.zeros((2*d, B, T), requires_grad=True).cuda()
		for ba in range(B):
			H_tilda[:,ba, :] = h_tilda[:,ba].repeat(T,1).transpose_(0,1)

		### Query aware representation of the context words
		#print('Query Aware representation of the context words')
		G = torch.zeros((8*d,B, T),requires_grad=True).cuda()
		for i in range(T):
			for ba in range(B):
				input_data = torch.cat([H[i,ba, :], U_tilda[:,ba,i], H[i,ba,:]*U_tilda[:,ba,i], H[i,ba,:]*H_tilda[:,ba,i]])
				G[:,ba,i] = self.att_model1(input_data)
		#print('\n\n\n\n\\n\n')
		#G = G.reshape([T,B,d])

		G = G.view([T,B,8*d])

		M,_ = self.bi_lstm2(G, None)
		M1 = M
		#M1=(M.reshape(M.shape[0], M.shape[2])).transpose_(0, 1)

		#################### Output layer : application specific layer
		## inputs: G, M1, M2
		## outputs : p1 and p2 - start and end indices of the answers respectively
		#print('Output Layer')
		M2,_ = self.bi_lstm3(M, None)
		#M2= (M2.reshape(M2.shape[0], M2.shape[2])).transpose_(0, 1)
		p1 = torch.zeros([T,B],requires_grad=True).cuda()
		p2 = torch.zeros([T,B],requires_grad=True).cuda()
		try:
			for ba in range(B):
				for t in range(T):
					p1[t, ba] = self.out_layer_1(torch.cat([G,M1],2)[t,ba,:]    )
					p2[t, ba] = self.out_layer_2(torch.cat([G,M2],2)[t,ba,:]    )

			p1_ = nn.functional.softmax(p1, dim=1)
			p2_ = nn.functional.softmax(p2, dim=1)
			return p1_, p2_
		except Exception:
			print('This wasnt supposed to be printed out - something is wrong!!!!')
			pdb.set_trace()
