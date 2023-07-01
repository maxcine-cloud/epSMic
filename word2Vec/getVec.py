#!/usr/bin/env python
# coding: utf-8

"""
reference:Zulfiqar, Hasan, Zi-Jie Sun, Qin-Lai Huang, Shi-Shi Yuan, Hao Lv, Fu-Ying Dao, Hao Lin, and Yan-Wen Li. "Deep-4mCW2V: A sequence-based predictor to identify N4-methylcytosine sites in Escherichia coli." Methods (2021), doi: 10.1016/j.ymeth.2021.07.011.
"""
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import pandas as pd
import numpy as np
import os
import sys
import math
import random
import warnings
from sklearn import preprocessing
import sklearn.preprocessing
from gensim import corpora, models, similarities
import os
data_path=os.path.split(os.path.realpath(__file__))[0]+os.sep

trainname='mut_after_train'
testname='mut_after_test'  
testID='mut_after_testID'
trainID='mut_after_trainID'

def DNA2Sentence(dna, K):
	sentence = ""
	length = len(dna)
	for i in range(length - K + 1):
		if 'N' in dna[i: i + K]:
			sentence += 'N'*K + " "
		else:	
			sentence += dna[i: i + K] + " "
	#delete extra space
	sentence = sentence[0 : len(sentence) - 1]
	return sentence

def Get_Unsupervised(fname,gname,kmer):
	f = open(fname,'r')
	g = open(gname,'w')
	k = kmer
	for i in f:
		if '>' not in i:
			i = i.strip('\n').upper()
			line = DNA2Sentence(i,k)
			g.write(line+'\n')
	f.close()
	g.close()

def getWord_model(num_features,min_count):
	word_model = ""
	# print(os.path.abspath(os.path.dirname(os.path.dirname(__file__)))+os.sep)#+"word_vec_model"+str(mer)
	# print(os.path.isfile("word_vec_model"+str(mer)))
	if not os.path.isfile(data_path+"word_vec_model"+str(mer)):
		Get_Unsupervised(data_path+trainname+'.txt','all_worVec'+str(mer)+'.txt',mer)
		sentence = LineSentence(data_path+'all_worVec'+str(mer)+'.txt')
		print ("Start Training Word2Vec model...")
		# Set values for various parameters
		num_features = int(num_features)
		min_word_count = int(min_count)	 
		num_workers = 20		
		context = 20			
		downsampling = 1e-3	 # Downsample setting for frequent words

		# Initialize and train the model
		print ("Training Word2Vec model...")
		word_model = Word2Vec(sentence, workers=num_workers, vector_size=num_features, min_count=min_word_count, window=context, sample=downsampling, seed=1,epochs = 50)
		word_model.init_sims(replace=False)
		word_model.save(data_path+'word_vec_model'+str(mer))
		#print word_model.most_similar("CATAGT")
	else:
		print ("Loading Word2Vec model...")
		word_model = Word2Vec.load(data_path+'word_vec_model'+str(mer))
		#word_model.init_sims(replace=True)
	return word_model


def getAvgFeatureVecs(DNAdata1,model,num_features):
	counter = 0
	DNAFeatureVecs = np.zeros((len(DNAdata1),num_features), dtype="float32")
	for DNA in DNAdata1:
		if counter % 1000 == 0:
			print ("DNA %d of %d\r" % (counter, len(DNAdata1)))
			sys.stdout.flush()

		DNAFeatureVecs[counter][0:num_features] = np.mean(model.wv[DNA],axis = 0)
		counter += 1
	print()
	counter = 0
	return DNAFeatureVecs


def getDNA_split(DNAdata,word):
	DNAlist1 = []
	#DNAlist2 = []
	for i in range(0,len(DNAdata)):
		# print(DNA)
		if '>' not in str(DNAdata.iloc[i,0]):
			# DNA = str(DNA).upper()
			DNAlist1.append(DNA2Sentence(DNAdata.iloc[i,0],word).split(" "))
	return DNAlist1


train=pd.read_csv(data_path+trainname+'.vcf',sep='\t',header=None)

mer=2
datawords1 = getDNA_split(train,mer)
word_model = getWord_model(200,1)
dataDataVecs = getAvgFeatureVecs(datawords1,word_model,200)
print(len(dataDataVecs))
id_train=pd.read_csv(data_path+trainID+'.vcf',sep='\t',header=None)
print(len(id_train))
np.save(data_path+"train.npy",np.hstack((id_train,dataDataVecs)))

test=pd.read_csv(data_path+testname+'.vcf',sep='\t',header=None)
datawords2 = getDNA_split(test,mer)
dataDataVecs2 = getAvgFeatureVecs(datawords2,word_model,200)
id_test=pd.read_csv(data_path+testID+'.vcf',sep='\t',header=None)
np.save(data_path+"test.npy", np.hstack((id_test,dataDataVecs2)))
