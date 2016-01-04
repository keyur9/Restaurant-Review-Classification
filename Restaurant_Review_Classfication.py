# coding: utf-8

# @author: Keyur Doshi
# Final Project: Restaurant Review Classification

#    Your script should read a file called "training.txt" and a file called "testing.txt". 
#    
#    The first file includes 1 review per line and 2 tab-separated columns. The first column is the review text and the second column is the label (1 for positive 0 for negative). 
#    
#    The second file includes 1 review per line. The line includes only the text of the review, but not the label. 
#    
#    Your script should predict the label for each of the reviews in "testing.txt". It should then write the labels to a file called "out.txt". Write one label per line (1 for positive 0 for negative), as follows: 
#    
#    0 
#    1 
#    0 
#    1 
#    1 
#    0 
#    
#    Your script should be done in 7 minutes or less. 
#    Each team can submit 10 times every 24 hours.
# In[246]:

# Import Library
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier

# In[247]:

#read the reviews and their polarities from a given file
def loadData(fname):
    reviews=[]
    labels=[]
    f=open(fname)
    for line in f:
        review,rating=line.strip().split('\t')  
        reviews.append(review.lower())    
        labels.append(int(rating))
    f.close()
    return reviews,labels
    
def loadtestData(fname):
    datas=[]
    f=open(fname)
    for data in f:
        datas.append(data.lower().strip())    
    f.close()
    return datas

rev_train,labels_train=loadData('training.txt')
rev_test=loadtestData('testing.txt')

# In[251]:

# Process data
def ProcessData(train_text):
    #x = 0
    output_arr = []
    for i in train_text:
        i = re.sub("\d+", "", i)
        i = re.sub(r"http\S+", "", i) #re.sub(r'^http?:\/\/.*[\r\n]*', '', train_text[i]) # remove hyperlinks
        i = re.sub('\.{2,}','. ', i)
        i = re.sub('<[a-zA-Z]*>', ' ', i)
        i = i.replace('&#;', '\'')
        i=re.sub(r'\'d',' would ',i) #replace 'd to would
        i = re.sub(r'\'t','ot ', i)#replace abbreviation #n
        i=re.sub(r'\'s','',i) #remove 's 
        i=re.sub(r'\'m',' am ',i) #replace 'm to am
        i=re.sub(r'\'re',' are ',i) #replace 're to are
        i=re.sub(r'\'ve',' have ',i) #replace 've to have
        i=re.sub(r'&amp','',i)
        i=re.sub(r'[?|$|.|!|( | ) | , | ; | " | / | - | : | % | -- ]',r' ',i) #| \'
        i=re.sub(' +',' ',i) #remove duplicate spaces
        i = i.encode('utf8')  # decoding data 
        i = i.lower() #lower   
        output_arr.append(i)
    return output_arr
    

# In[253]:

# Combine not with next word. eg. This is not bad --> This is notbad
def CombineNot(train_text):
    output_arr = []
    for sentence in train_text:
        input_words = sentence.split(" ")
        output_str = input_words[0]
        i = 0
        while(i < len(input_words)-1):
            #print "i = " + str(i) + input_words[i]
            if input_words[i] == "not":
                output_str += " " + input_words[i] + input_words[i+1]
            else :
                output_str += " " + input_words[i] + " " + input_words[i+1]
            i+=2
        output_arr.append(output_str)
    return output_arr

# In[254]:

# Preprocessing and combining on training data
rev_train = ProcessData(rev_train)
rev_train = CombineNot(rev_train)

# Preprocessing and combining on testing data
rev_test=ProcessData(rev_test)
rev_test=CombineNot(rev_test)

# In[]

# Import and initialize Stemmer and Lemmatizer
import nltk
porter = nltk.PorterStemmer()
lmtzr = nltk.stem.wordnet.WordNetLemmatizer()


# Normalized test data
normalized_test=[]
for token in rev_test:
    rev_test_small=[]
    for item in token.strip().split(' '):
        if (len(item) >= 2):
            rev_test_small.append(lmtzr.lemmatize(porter.stem(item).encode('utf8'))) 
    normalized_test.append((rev_test_small)) 

k = 0
final_test_normalized=[]
for k in range(len(normalized_test)):
    final_test_normalized.append(' '.join(normalized_test[k]))
    k += 1

# Normalized train data
normalized_train=[]
for tokens in rev_train:
    rev_train_small=[]
    for items in tokens.strip().split(' '):
        if (len(items) >= 2):
            rev_train_small.append(lmtzr.lemmatize(porter.stem(items).encode('utf8'))) 
    normalized_train.append((rev_train_small)) 

q = 0
final_train_normalized=[]
for q in range(len(normalized_train)):
    final_train_normalized.append(' '.join(normalized_train[q]))
    q += 1

# In[255]:

# Analysis Data
# Build a counter based on the training dataset
counter = CountVectorizer(analyzer = 'word', stop_words=['you','yours','ours','a','an','the','i','we','our','us','today','tomorrow','week','month','year','yesterday','weekend','weeknight','day','afternoon','Saturday','Sunday','night','day'
                    , 'Canadian','English', 'weekday','Mexican','Seattle'],ngram_range=(1,3),max_features=70000) #1055
counter.fit(final_train_normalized) #only focus on the text appear in rev_train

# In[256]:

#count the number of times each term appears in a document and transform each doc into a count vector
counts_train = counter.transform(final_train_normalized)#transform the training data
counts_test = counter.transform(final_test_normalized)#transform the testing data

# In[257]:

# Classifier to predict the reviews
clf1 = LogisticRegression(tol = 0.01)
clf2 = KNeighborsClassifier(3) 
clf3 = MultinomialNB(alpha = 1.0)


# In[258]:

# Creating a Voting team
classifier = VotingClassifier(estimators=[('k',clf2), ('lr', clf1), ('mnb', clf3)], voting='soft',weights = [1,1,2])

# In[259]:

# Use the classifier to fit 
classifier.fit(counts_train, labels_train)


# In[260]:


# use the classifier to predict
predicted=classifier.predict(counts_test)


# In[261]:

# Check accuracy in local
'''
from sklearn import metrics
print metrics.accuracy_score(predicted, labels_test)
print(metrics.zero_one_loss(labels_test, predicted,normalize = False))
print metrics.precision_score(labels_test, predicted)
print metrics.recall_score(labels_test, predicted)
'''
# In[ ]:

# Writing the output to the file
resultwriter=open('out.txt','w')
for each_predicted in predicted:
    resultwriter.write(str(each_predicted)+'\n')
resultwriter.close()

# In[]:

# EXTRA
# Write the predicted label and compare it against original label in a tabular format

'''
filewriter = open('pt','w')
from prettytable import PrettyTable
t = PrettyTable(['Predicted','Expected'])
#t.add_row([labels_test,predicted])
for i in range(len(predicted)):
    t.add_row([predicted[i],labels_test[i]])
    i = i + 1

filewriter.write(str(t))
filewriter.close()
'''

