from sklearn import model_selection

import pandas as pd 
from pandas import DataFrame, read_csv
import os
import csv 
import numpy as np 
from pandas.tools.plotting import scatter_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score,roc_curve,brier_score_loss,precision_recall_curve
import matplotlib.pyplot as plt

train_path = "aclImdb/ALL/" # source data
test_path = "test/imdb_te.csv" # test data for grade evaluation. 
#kfold = model_selection.KFold(n_splits=10, random_state=7)
'''
IMDB_DATA_PREPROCESS explores the neg and pos folders from aclImdb/train and creates a output_file in the required format
Inpath - Path of the training samples 
Outpath - Path were the file has to be saved 
Name  - Name with which the file has to be saved 
Mix - Used for shuffling the data 
'''
def imdb_data_preprocess(inpath, outpath="./", name="imdb_tr.csv", mix=False):
	

	stopwords = open("stopwords.en.txt", 'r' , encoding="ISO-8859-1").read()
	stopwords = stopwords.split("\n")

	indices = []
	text = []
	rating = []

	i =  0 

	for filename in os.listdir(inpath+"pos"):
		data = open(inpath+"pos/"+filename, 'r' , encoding="ISO-8859-1").read()
		data = remove_stopwords(data, stopwords)
		indices.append(i)
		text.append(data)
		rating.append("1")
		i = i + 1

	for filename in os.listdir(inpath+"neg"):
		data = open(inpath+"neg/"+filename, 'r' , encoding="ISO-8859-1").read()
		data = remove_stopwords(data, stopwords)
		indices.append(i)
		text.append(data)
		rating.append("0")
		i = i + 1

	Dataset = list(zip(indices,text,rating))
	
	if mix:
		np.random.shuffle(Dataset)

	df = pd.DataFrame(data = Dataset, columns=['row_Number', 'text', 'polarity'])
	df.to_csv(outpath+name, index=False, header=True)

	pass


'''
REMOVE_STOPWORDS takes a sentence and the stopwords as inputs and returns the sentence without any stopwords 
Sentence - The input from which the stopwords have to be removed
Stopwords - A list of stopwords  
'''
def remove_stopwords(sentence, stopwords):
	sentencewords = sentence.split()
	resultwords  = [word for word in sentencewords if word.lower() not in stopwords]
	result = ' '.join(resultwords)
	return result


'''
UNIGRAM_PROCESS takes the data to be fit as the input and returns a vectorizer of the unigram as output 
Data - The data for which the unigram model has to be fit 
'''
def unigram_process(data):
	from sklearn.feature_extraction.text import CountVectorizer
	vectorizer = CountVectorizer()
	vectorizer = vectorizer.fit(data)
	return vectorizer	


'''
BIGRAM_PROCESS takes the data to be fit as the input and returns a vectorizer of the bigram as output 
Data - The data for which the bigram model has to be fit 
'''
def bigram_process(data):
	from sklearn.feature_extraction.text import CountVectorizer
	vectorizer = CountVectorizer(ngram_range=(1,2))
	vectorizer = vectorizer.fit(data)
	return vectorizer


'''
TFIDF_PROCESS takes the data to be fit as the input and returns a vectorizer of the tfidf as output 
Data - The data for which the bigram model has to be fit 
'''
def tfidf_process(data):
	from sklearn.feature_extraction.text import TfidfTransformer 
	transformer = TfidfTransformer()
	transformer = transformer.fit(data)
	return transformer


'''
RETRIEVE_DATA takes a CSV file as the input and returns the corresponding arrays of labels and data as output. 
Name - Name of the csv file 
Train - If train is True, both the data and labels are returned. Else only the data is returned 
'''
def retrieve_data(name="imdb_tr.csv", train=True):
	import pandas as pd 
	data = pd.read_csv(name,header=0, encoding = 'ISO-8859-1')
	X = data['text']
	
	if train:
		Y = data['polarity']
		return X, Y

	return X		


'''
Bagging : Random Forest Classifier on the training data and returns the predicted labels 
Xtrain - Training Data
Ytrain - Training Labels
Xtest - Test Data 
'''
def random_forest_classifier(Xtrain, Ytrain, Xtest):
	from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
	from sklearn.ensemble import BaggingClassifier
	from sklearn.tree import DecisionTreeClassifier

	clf = RandomForestClassifier()

	print ("Random Forest Fitting")
	clf.fit(Xtrain, Ytrain)
	print ("Random Forest Predicting")
	Ytest = clf.predict(Xtest)

	return Ytest


'''
Bagging : Extra Tree Classifier on the training data and returns the predicted labels 
Xtrain - Training Data
Ytrain - Training Labels
Xtest - Test Data 
'''
def extra_tree_classifier(Xtrain, Ytrain, Xtest):
	from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
	from sklearn.ensemble import BaggingClassifier
	from sklearn.tree import DecisionTreeClassifier

	clf = ExtraTreesClassifier()
	print ("Extra Tree Fitting")
	clf.fit(Xtrain, Ytrain)
	print ("Extra Tree Predicting")
	Ytest = clf.predict(Xtest)
	return Ytest



'''
Bagging : Bagged Decision Tree Classifier on the training data and returns the predicted labels 
Xtrain - Training Data
Ytrain - Training Labels
Xtest - Test Data 
'''
def bagging_decision_tree(Xtrain, Ytrain, Xtest):
	from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
	from sklearn.ensemble import BaggingClassifier
	from sklearn.tree import DecisionTreeClassifier

	clf = BaggingClassifier(base_estimator=DecisionTreeClassifier())
	print ("Bagged Decision Tree Fitting")
	clf.fit(Xtrain, Ytrain)
	print ("Bagged Decision Tree Predicting")
	Ytest = clf.predict(Xtest)
	return Ytest

'''
Boosting : Ada Boost Classifier on the training data and returns the predicted labels 
Xtrain - Training Data
Ytrain - Training Labels
Xtest - Test Data 
'''
def ada_boost_classifier(Xtrain, Ytrain, Xtest):
	from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
	from sklearn.ensemble import BaggingClassifier
	from sklearn.tree import DecisionTreeClassifier

	clf = AdaBoostClassifier()
	print ("Ada Boost Classifier Fitting")
	clf.fit(Xtrain, Ytrain)
	print ("Ada Boost Classifier Predicting")
	Ytest = clf.predict(Xtest)
	return Ytest



'''
Boosting : Gradient Boost Classifier on the training data and returns the predicted labels 
Xtrain - Training Data
Ytrain - Training Labels
Xtest - Test Data 
'''
def gradient_boost_classifier(Xtrain, Ytrain, Xtest):
	from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
	from sklearn.ensemble import BaggingClassifier
	from sklearn.tree import DecisionTreeClassifier

	clf = GradientBoostingClassifier()
	print ("Gradient Boost Classifier Fitting")
	clf.fit(Xtrain, Ytrain)
	print ("Gradient Boost Classifier Predicting")
	Ytest = clf.predict(Xtest)
	return Ytest

'''
ACCURACY finds the accuracy in percentage given the training and test labels 
Ytrain - One set of labels 
Ytest - Other set of labels 
'''
def accuracy(Ytrain, Ytest):
	assert (len(Ytrain)==len(Ytest))
	num =  sum([1 for i, word in enumerate(Ytrain) if Ytest[i]==word])
	n = len(Ytrain)  
	return (num*100)/n


if __name__ == "__main__":
	import time
	start = time.time()
	print ("Preprocessing the training_data--")
	imdb_data_preprocess(inpath=train_path, mix=True)
	print ("Done with preprocessing. Now, will retreieve the training data in the required format")
	[Xtrain_text, Ytrain] = retrieve_data()
	print ("Retrieved the training data. Now will retrieve the test data in the required format")
	Xtest_text = retrieve_data(name=test_path, train=False)
	print ("Retrieved the test data. Now will initialize the model \n\n")


	print ("-----------------------ANALYSIS ON THE INSAMPLE DATA---------------------------")
	uni_vectorizer = unigram_process(Xtrain_text)
	print ("Fitting the unigram model")
	Xtrain_uni = uni_vectorizer.transform(Xtrain_text)
	print ("After fitting ")
	
	
	print ("\n")
	print ("\n")
	print ("-------------------Ensembe : Bagging --------------------")
	print ("-------------------Random Forest Report--------------------")
	print ("Applying the Random Forest")
	Y_uni = random_forest_classifier(Xtrain_uni, Ytrain, Xtrain_uni)
	print ("Done with  Random Forest")
	print ("Accuracy for the Unigram Model is ", accuracy(Ytrain, Y_uni))
	
	
	print (confusion_matrix(Ytrain, Y_uni))
	print (classification_report(Ytrain, Y_uni))

	print ("\n")
	print ("-------------------Extra Tree Report ---------------------")
	print ("Applying the Extra Tree")
	Y_uni = extra_tree_classifier(Xtrain_uni, Ytrain, Xtrain_uni)
	print ("Done with  Extra Tree")
	print ("Accuracy for the Unigram Model is ", accuracy(Ytrain, Y_uni))
	
	print (confusion_matrix(Ytrain, Y_uni))
	print (classification_report(Ytrain, Y_uni))
	print ("\n")
	print ("-------------------Bagged Decision Tree Report ---------------------")
	print ("Applying the Bagged Decision Tree")
	Y_uni = bagging_decision_tree(Xtrain_uni, Ytrain, Xtrain_uni)
	print ("Done with  Bagged Decision Tree")
	print ("Accuracy for the Unigram Model is ", accuracy(Ytrain, Y_uni))
	print (confusion_matrix(Ytrain, Y_uni))
	print (classification_report(Ytrain, Y_uni))
	print ("\n")
	print ("\n")

	print ("-------------------Ensembe : Boosting --------------------")
	print ("-------------------Ada Boost Report ---------------------")
	print ("Applying the Ada Boost")
	Y_uni = ada_boost_classifier(Xtrain_uni, Ytrain, Xtrain_uni)
	print ("Done with  Ada Boost")
	print ("Accuracy for the Unigram Model is ", accuracy(Ytrain, Y_uni))
	print (confusion_matrix(Ytrain, Y_uni))
	print (classification_report(Ytrain, Y_uni))
	print ("\n")
	print ("-------------------Gradient Boost Report - Train Data--------------------")
	print ("Applying the Gradient Boost")
	Y_uni = gradient_boost_classifier(Xtrain_uni, Ytrain, Xtrain_uni)
	print ("Done with  Gradient Boost ")
	print ("Accuracy for the Unigram Model is ", accuracy(Ytrain, Y_uni))
	print (confusion_matrix(Ytrain, Y_uni))
	print (classification_report(Ytrain, Y_uni))
	print ("\n")
	print ("\n")

	bi_vectorizer = bigram_process(Xtrain_text)
	print ("Fitting the bigram model")
	Xtrain_bi = bi_vectorizer.transform(Xtrain_text)
	print ("After fitting ")
	
	print ("\n")
	print ("\n")

	print ("-------------------Ensembe : Bagging --------------------")
	print ("-------------------Random Forest Report --------------------")
	print ("Applying the Random Forest")
	Y_bi = random_forest_classifier(Xtrain_bi, Ytrain, Xtrain_bi)
	print ("Done with  Random Forest")
	print ("Accuracy for the Bigram Model is ", accuracy(Ytrain, Y_bi))


	print (confusion_matrix(Ytrain, Y_bi))
	print (classification_report(Ytrain, Y_bi))


	print ("\n")
	print ("-------------------Extra Tree Report--------------------")
	print ("Applying the Extra Tree")
	Y_bi = extra_tree_classifier(Xtrain_bi, Ytrain, Xtrain_bi)
	print ("Done with  Extra Tree")
	print ("Accuracy for the Bigram Model is ", accuracy(Ytrain, Y_bi))

	print (confusion_matrix(Ytrain, Y_bi))
	print (classification_report(Ytrain, Y_bi))

	print ("\n")
	print ("-------------------Bagged Decision Tree Report--------------------")
	print ("Applying the Bagged Decision Tree")
	Y_bi = bagging_decision_tree(Xtrain_bi, Ytrain, Xtrain_bi)
	print ("Done with  Bagged Decision Tree")
	print ("Accuracy for the Bigram Model is ", accuracy(Ytrain, Y_bi))
	print (confusion_matrix(Ytrain, Y_bi))
	print (classification_report(Ytrain, Y_bi))

	print ("\n")
	print ("\n")
	print ("-------------------Ensembe : Boosting --------------------")
	print ("-------------------Ada Boost Report ---------------------")
	print ("Applying the Ada Boost")
	Y_bi = ada_boost_classifier(Xtrain_bi, Ytrain, Xtrain_bi)
	print ("Done with  Ada Boost")
	print ("Accuracy for the Bigram Model is ", accuracy(Ytrain, Y_bi))
	print (confusion_matrix(Ytrain, Y_bi))
	print (classification_report(Ytrain, Y_bi))

	print ("\n")
	print ("-------------------Gradient Boost Report ---------------------")
	print ("Applying the Gradient Boost")
	Y_bi = gradient_boost_classifier(Xtrain_bi, Ytrain, Xtrain_bi)
	print ("Done with  Gradient Boost ")
	print ("Accuracy for the Bigram Model is ", accuracy(Ytrain, Y_bi))
	print (confusion_matrix(Ytrain, Y_bi))
	print (classification_report(Ytrain, Y_bi))

	print ("\n")

	print ("\n")


	uni_tfidf_transformer = tfidf_process(Xtrain_uni)
	print ("Fitting the tfidf for unigram model")
	Xtrain_tf_uni = uni_tfidf_transformer.transform(Xtrain_uni)
	print ("After fitting TFIDF")
	
	print ("\n")
	print ("\n")
	print ("-------------------Ensembe : Bagging --------------------")
	print ("-------------------Random Forest Report --------------------")
	print ("Applying the Random Forest")
	Y_tf_uni = random_forest_classifier(Xtrain_tf_uni, Ytrain, Xtrain_tf_uni)
	print ("Done with  Random Forest")
	print ("Accuracy for the Unigram TFIDF Model is ", accuracy(Ytrain, Y_tf_uni))
	print (confusion_matrix(Ytrain, Y_tf_uni))
	print (classification_report(Ytrain, Y_tf_uni))
	print ("\n")
	print ("-------------------Extra Tree Report --------------------")
	print ("Applying the Extra Tree")
	Y_tf_uni = extra_tree_classifier(Xtrain_tf_uni, Ytrain, Xtrain_tf_uni)
	print ("Done with  Extra Tree")
	print ("Accuracy for the Unigram TFIDF Model is ", accuracy(Ytrain, Y_tf_uni))
	print (confusion_matrix(Ytrain, Y_tf_uni))
	print (classification_report(Ytrain, Y_tf_uni))
	print ("\n")
	print ("-------------------Bagged Decision Tree Report--------------------")
	print ("Applying the Bagged Decision Tree")
	Y_tf_uni = bagging_decision_tree(Xtrain_tf_uni, Ytrain, Xtrain_tf_uni)
	print ("Done with  Bagged Decision Tree")
	print ("Accuracy for the Unigram TFIDF Model is ", accuracy(Ytrain, Y_tf_uni))
	print (confusion_matrix(Ytrain, Y_tf_uni))
	print (classification_report(Ytrain, Y_tf_uni))
	print ("\n")
	print ("\n")

	print ("-------------------Ensembe : Boosting --------------------")
	print ("-------------------Ada Boost Report ---------------------")
	print ("Applying the Ada Boost")
	Y_tf_uni = ada_boost_classifier(Xtrain_tf_uni, Ytrain, Xtrain_tf_uni)
	print ("Done with  Ada Boost")
	print ("Accuracy for the Unigram TFIDF Model is ", accuracy(Ytrain, Y_tf_uni))
	print (confusion_matrix(Ytrain, Y_tf_uni))
	print (classification_report(Ytrain, Y_tf_uni))
	print ("\n")
	print ("-------------------Gradient Boost Report--------------------")
	print ("Applying the Gradient Boost")
	Y_tf_uni = gradient_boost_classifier(Xtrain_tf_uni, Ytrain, Xtrain_tf_uni)
	print ("Done with  Gradient Boost ")
	print ("Accuracy for the Unigram TFIDF Model is ", accuracy(Ytrain, Y_tf_uni))
	print (confusion_matrix(Ytrain, Y_tf_uni))
	print (classification_report(Ytrain, Y_tf_uni))
	print ("\n")

	print ("\n")


	bi_tfidf_transformer = tfidf_process(Xtrain_bi)
	print ("Fitting the tfidf for bigram model")
	Xtrain_tf_bi = bi_tfidf_transformer.transform(Xtrain_bi)
	print ("After fitting TFIDF")


	print ("\n")
	print ("\n")
	print ("-------------------Ensembe : Bagging --------------------")
	print ("-------------------Random Forest Report--------------------")
	print ("Applying the Random Forest")
	Y_tf_bi = random_forest_classifier(Xtrain_tf_bi, Ytrain, Xtrain_tf_bi)
	print ("Done with  Random Forest")
	print ("Accuracy for the Bigram TFIDF Model is ", accuracy(Ytrain, Y_tf_bi))
	print (confusion_matrix(Ytrain, Y_tf_bi))
	print (classification_report(Ytrain, Y_tf_bi))
	print ("\n")
	print ("-------------------Extra Tree Report --------------------")
	print ("Applying the Extra Tree")
	Y_tf_bi = extra_tree_classifier(Xtrain_tf_bi, Ytrain, Xtrain_tf_bi)
	print ("Done with  Extra Tree")
	print ("Accuracy for the Bigram TFIDF Model is ", accuracy(Ytrain, Y_tf_bi))
	print (confusion_matrix(Ytrain, Y_tf_bi))
	print (classification_report(Ytrain, Y_tf_bi))
	print ("\n")
	print ("-------------------Bagged Decision Tree Report --------------------")
	print ("Applying the Bagged Decision Tree")
	Y_tf_bi = bagging_decision_tree(Xtrain_tf_bi, Ytrain, Xtrain_tf_bi)
	print ("Done with  Bagged Decision Tree")
	print ("Accuracy for the Bigram TFIDF Model is ", accuracy(Ytrain, Y_tf_bi))
	print (confusion_matrix(Ytrain, Y_tf_bi))
	print (classification_report(Ytrain, Y_tf_bi))
	print ("\n")
	print ("\n")

	print ("-------------------Ensembe : Boosting --------------------")
	print ("-------------------Ada Boost Report --------------------")
	print ("Applying the Ada Boost")
	Y_tf_bi = ada_boost_classifier(Xtrain_tf_bi, Ytrain, Xtrain_tf_bi)
	print ("Done with  Ada Boost")
	print ("Accuracy for the Bigram TFIDF Model is ", accuracy(Ytrain, Y_tf_bi))
	print (confusion_matrix(Ytrain, Y_tf_bi))
	print (classification_report(Ytrain, Y_tf_bi))
	print ("\n")
	print ("-------------------Gradient Boost Report--------------------")
	print ("Applying the Gradient Boost")
	Y_tf_bi = gradient_boost_classifier(Xtrain_tf_bi, Ytrain, Xtrain_tf_bi)
	print ("Done with  Gradient Boost ")
	print ("Accuracy for the Bigram TFIDF Model is ", accuracy(Ytrain, Y_tf_bi))
	print (confusion_matrix(Ytrain, Y_tf_bi))
	print (classification_report(Ytrain, Y_tf_bi))

	print ("\n")

	print ("\n")




	print ("Total time taken is ", time.time()-start, " seconds")
	pass
