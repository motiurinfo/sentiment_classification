# sentiment_classification
Performance evaluation of sentiment classification in movie reviews

## 1. Introduction ##

Given the availability of a large volume of online review data (Amazon, IMDB, etc.), sentiment analysis becomes increasingly important. In this project, a sentiment sentiment classification is evaluated using ensemble methods. 

## 2. Getting the Dataset ##
This can also be downloaded from: http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz. 

## 3. Data Preprocessing ##

The training dataset in aclImdb folder has two sub-directories pos/ for positive texts and neg/ for negative ones. Use only these two directories. The first task is to combine both of them to a single csv file, “imdb_tr.csv”. The csv file has three columns,"row_number" and “text” and “polarity”. The column “text” contains review texts from the aclImdb database and the column “polarity” consists of sentiment labels, 1 for positive and 0 for negative. The file imdb_tr.csv is an output of this preprocessing. In addition, common English stopwords should be removed. An English stopwords reference ('stopwords.en') is given in the code for reference.


## 4. Data Representations Used  ##

Vectorization methods: Unigram , Bigram
Feature Extraction: TfIdf 


## 5. Algorithmic Overview ##

In this project, we will train ensemble methods and evaluate the optimized combination:

http://scikit-learn.org/stable/modules/ensemble.html

## 6. Functions used in the sentimentalAnalysis file ##

imdb_data_preprocess : Explores the neg and pos folders from aclImdb/train and creates a imdb_tr.csv file in the required format

remove_stopwords : Takes a sentence and the stopwords as inputs and returns the sentence without any stopwords

unigram_process : Takes the data to be fit as the input and returns a vectorizer of the unigram as output

bigram_process : Takes the data to be fit as the input and returns a vectorizer of the bigram as output 

tfidf_process : Takes the data to be fit as the input and returns a vectorizer of the tfidf as output

retrieve_data : Takes a CSV file as the input and returns the corresponding arrays of labels and data as output

random_forest_classifier : Applies Random Forest on the training data and returns the predicted labels

extra_tree_classifier : Applies Extra Tree on the training data and returns the predicted labels

bagging_decision_tree : Applies Bagged Decision Tree on the training data and returns the predicted labels

ada_boost_classifier : Applies ADA Boost on the training data and returns the predicted labels

gradient_boost_classifier : Applies Gradient Boost on the training data and returns the predicted labels

accuracy : Finds the accuracy in percentage given the training and test labels

## 7. Environment ##

OS: Linux Mint

Language : Python 3

Libraries : Scikit, Pandas 

## 8. How to Execute? ##

Run python sentimentalAnalysis.py

## 9. Screenshots ##
Check Result in screenshots folder

## 10. Publication ##
## Paper Title: ##
Supervised Ensemble Machine Learning Aided Performance Evaluation of Sentiment Classification
## Authonrs: ##
Sheikh Shah Mohammad Motiur Rahman,Md. Habibur Rahman,Kaushik Sarker,Md. Samadur Rahman, Nazmul Ahsan,M. Mesbahuddin Sarker
## Conference Info: ##
2nd International Conference on Data Mining, Communications and Information Technology (DMCIT 2018), Shanghai, China
