import os
#os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
import random
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
from tqdm import tqdm
import pickle
import string
from tmu.models.classification.vanilla_classifier import TMClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.metrics import f1_score
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))
import numpy as np
np.random.seed(400) 
from time import time


dataset = 'TREC'
doc_name_list = []
doc_train_list = []
doc_test_list = []

sizes=[1,2,3,4]

def remove_stopwords(words):
    words = [word for word in words if not word in stopwords]
    return words

def tokenize_line(line):
    words= line[0:len(line)-1].strip().split(" ")
    text= remove_stopwords(words)
    return text

def flatten(list):
    new_list = []
    for i in list:
        for j in i:
            new_list.append(j)
    return new_list

for scaling in sizes:

    hypervector_size = 3000*scaling
    
    #p = 0.25*1.0/np.sqrt(hypervector_size)
    p = 0.0005
    bits = int(p * hypervector_size)
    
    FEATURES=3000
    
    print("Producing encoding...")
    encoding = np.zeros((FEATURES, hypervector_size), dtype=np.uint32)
    indexes = np.arange(hypervector_size, dtype=np.uint32)
    for i in range(FEATURES):
    	selection = np.random.choice(indexes, size=(bits))
    	encoding[i][selection] = 1
    hash_value =  np.zeros(hypervector_size, dtype=np.uint32)
    
    for i in range(FEATURES):
    	one_ids = encoding[i].nonzero()[0]
    	hash_value[i] = hash(one_ids.tobytes()) % (2 ** 29 - 1)
    
    with open('./' + dataset + '.txt', 'r') as f:
        for line in f.readlines():
            doc_name_list.append(line.strip())
            temp = line.split("\t")
    
            if temp[1].find('test') != -1:
                doc_test_list.append(line.strip())
            elif temp[1].find('train') != -1:
                doc_train_list.append(line.strip())
    
    doc_content_list = []
    
    with open('./' + dataset + '.clean.txt', 'r') as f:
        for line in f.readlines():
            doc_content_list.append(line.strip())

    train_ids = []
    for train_name in doc_train_list:
        train_id = doc_name_list.index(train_name)
        train_ids.append(train_id)
    random.shuffle(train_ids)
    
    test_ids = []
    for test_name in doc_test_list:
        test_id = doc_name_list.index(test_name)
        test_ids.append(test_id)
    random.shuffle(test_ids)
    

    train_content=[]
    train_label=[]
    for i in train_ids:
        temp= doc_name_list[int(i)].split('\t')
        train_label.append(temp[2])
        train_content.append(tokenize_line(doc_content_list[int(i)]))
        
    test_content=[]
    test_label=[]
    for i in test_ids:
        temp= doc_name_list[int(i)].split('\t')
        test_label.append(temp[2])
        test_content.append(tokenize_line(doc_content_list[int(i)]))

    label_encoder = preprocessing.LabelEncoder()
    train_y = label_encoder.fit_transform(train_label)
    test_y= label_encoder.transform(test_label)

    def tokenizer(s):
    	return s
    
    vectorizer_X = CountVectorizer(tokenizer=tokenizer, lowercase=False, ngram_range=(1,2), binary=True)
    
    X_train= vectorizer_X.fit_transform(train_content)
    X_test = vectorizer_X.transform(test_content)

    print("Selecting features...")
    
    SKB = SelectKBest(chi2, k=FEATURES)
    SKB.fit(X_train, train_y)
    
    X_train_org = SKB.transform(X_train)
    X_test_org = SKB.transform(X_test)
    
    X_train = np.zeros((train_y.shape[0], hypervector_size), dtype=np.uint32)
    Y_train = np.zeros(train_y.shape[0], dtype=np.uint32)
    for i in range(train_y.shape[0]):
    	for word_id in X_train_org.getrow(i).indices:
    		X_train[i] = np.logical_or(X_train[i], encoding[word_id])
    	Y_train[i] = train_y[i]
    
    X_test = np.zeros((test_y.shape[0], hypervector_size), dtype=np.uint32)
    Y_test = np.zeros(test_y.shape[0], dtype=np.uint32)
    for i in range(test_y.shape[0]):
    	for word_id in X_test_org.getrow(i).indices:
    		X_test[i] = np.logical_or(X_test[i], encoding[word_id])
    	Y_test[i] = test_y[i]
    

    print(X_train.shape, Y_train.shape)
    NUM_CLAUSES=8000
    print("NUM_CLAUSES :",NUM_CLAUSES)
    threshold=[150]      
    SP= [ 10.0]     
    print("Training started")
    
    
    for S in SP:
        for THRESHOLD in threshold:
            f = open("result_trec_%.1f_%d_%d_%d.txt" % (S, NUM_CLAUSES, THRESHOLD, scaling), "w+")
            tm = TMClassifier(NUM_CLAUSES, THRESHOLD, S, platform='CUDA')
            print("\nAccuracy over 100 epochs:\n")
            time_diff=[]
            acc_train=[]
            acc_test=[]
            f1_score_test=[]
            for i in range(150):
                print(f'epochs: {i}')
                start_training = time()
                tm.fit(X_train, Y_train)
                stop_training = time()
                result = 100*(tm.predict(X_train) == Y_train).mean()
                
                start_testing = time()
                result_test = 100*(tm.predict(X_test) == Y_test).mean()
                stop_testing = time()
                
                f1= f1_score(Y_test, tm.predict(X_test), average='macro' )
                f1_score_test.append(round(f1,4))
                print(f'{result: .2f} {result_test: .2f} {f1:.2f} {stop_training-start_training: .2f} {stop_testing-start_testing: .2f} {scaling}', file=f)
                f.flush()
    f.close

