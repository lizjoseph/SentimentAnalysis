
# coding: utf-8

# In[1]:

import nltk
nltk.download('punkt')


# In[2]:

import nltk
nltk.download('stopwords')


# In[3]:

import itertools



# In[ ]:

from nltk.tokenize import word_tokenize
import re, string, timeit
import HTMLParser
from nltk.corpus import stopwords
import string
html_parser = HTMLParser.HTMLParser()

arrayofTweets = []

#start replaceTwoOrMore
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL) 
    return pattern.sub(r"\1\1", s)
#end

with open("C:\Anaconda2\NewDataSet.txt", "r") as ins:
    for line in ins:
        #Convert to lower case
        begintweet = line.lower()
        withoutescape = html_parser.unescape(begintweet)
        decodedtweet = withoutescape.decode("utf8").encode('ascii','ignore')
        word1 = decodedtweet.replace("&amp", "&")
        word2 = word1.replace("'s'", " is ")
        word3 = word2.replace("'ve", " have ")
        word4 = word3.replace("'re", " are ")
        word5 = word4.replace("n't", " not ")
        wordfive = word5.replace("'ll", " will ")
        wordsix = wordfive.replace("..", " ")
        word6 = wordsix.replace("\\", " ")
        word7 = word6.replace('"', ' ')
        #Convert www.* or https?://* to URL
        word8 = re.sub(r"http\S+", " ", word7)
        #Convert @username to AT_USER
        word9= re.sub('@[^\s]+',' ',word8)
        #Replace #word with word
        word10 =re.sub(r'\n\S+', ' ', word9)
        word11 = re.sub(r'#([^\s]+)', r' \1 ', word10)
        word12 = re.sub(r"\ud\S+", " ", word11)
        worded = re.sub(r"\u2\S+", " ", word12)
        worded1 = re.sub(r"\uf\S+", " ", worded)
        worded2 = re.sub(r"\\\S+", " ", worded1)
        worded4 = re.sub( 't.co[^\s]+'," ",worded2 ) 
        worded5 = re.sub('\n', ' ', worded4)
        s = re.sub(r'[^\w\s]',' ',worded5) 
        s2 = re.sub("\d", " ", s)
        
        #Remove additional white spaces
        word14= re.sub('[\s]+', ' ', s2)
        
        punctuation = list(string.punctuation)
        stop = stopwords.words('english') + punctuation + ['rt', 'via']
        
        terms_stop = [term for term in word_tokenize(word14) if term not in stop]
        arrayBefores = []
    
        for x in terms_stop:
            if (len(x) > 3 and (x !='sa') and(x !='da')):
                word13 = replaceTwoOrMore(x)
                arrayBefores.append(x)
        if(arrayBefores != []):
            arrayofTweets.append(arrayBefores)


print arrayofTweets


# In[ ]:


categories = []
isThere = False

def equal(a, b):
    # Ignore non-space and non-word characters
    regex = re.compile(r'[^\s\w]')
    return regex.sub('', a) == regex.sub('', b)

def remove_duplicates(numbers):
    newlist = []
    for number in numbers:
        if number not in newlist:
            newlist.append(number)
    return newlist


#-----------------------------------------------------------------------

for line in arrayofTweets:
    for word in line:
        isThere = False
        with open("C:\Anaconda2\synonymslist.txt") as openfile:
            for line2 in openfile:
                for part in line2.split(','):
                    if equal(word, part):
                        isThere = True
                        save = line2.split(',')[0]
                        categories.append(save)
                    break
                        
                
        if(isThere is False):
            with open("C:\Anaconda2\synonymslist.txt") as openfile:
                    for line2 in openfile:
                        for part in line2.split():
                            if equal(word, part):
                                isThere = True
                                save = line2.split(',')[0]
                                categories.append(save)
                            break
            
            if(isThere is False):
                wordPut = word
                categories.append(wordPut)
            
                
categoriesNew = []
categoriesNew = remove_duplicates(categories)

print categoriesNew



# In[10]:

categorysize  =  (len(categoriesNew) + 3)
counter = 0

categoryPos = categorysize - 3
categoryNeg = categorysize - 2
categoryNeu = categorysize - 1

#print categorysize
#print "This is category vector size"

listOfVectors= []

for i in range(categorysize):
        vector = [0.0] * categorysize

def findIndex(cat):
    foundIndex = categoriesNew.index(cat) if cat in categoriesNew else -1
    return foundIndex

def checkSentiment(sentiment):
    sent = sentiment
    if sent == 'Positive':
        return categoryPos
    elif sent =='Negative':
        return categoryNeg
    else:
        return categoryNeu
    
vector = []
isThere = False

for tweets in arrayofTweets:
    vector = [0.0] * categorysize
    for word in tweets:
        isThere = False
        with open("C:\Anaconda2\synonymslist.txt") as openfile:
            for line2 in openfile:
                for part in line2.split(','):
                    if equal(word, part):
                        isThere = True
                        cat = line2.split(',')[0]
                        sentiment = line2.split(',')[-1]
                        x = findIndex(cat)
                        vector[x] = vector[x] + 1.0
                        sentInd = checkSentiment(sentiment)
                        vector[sentInd] = vector[sentInd] + 1.0
                    break           
                
                        
                        
        if(isThere is False):
            x = findIndex(word)
            if (x != -1):
                vector[x] = vector[x] + 1.0
                vector[categoryNeu] = vector[categoryNeu] + 1.0
                counter = counter + 1
               
        
                
            if (x == -1):
                with open("C:\Anaconda2\synonymslist.txt") as openfile:
                    for line2 in openfile:
                        for part in line2.split():
                            if equal(word, part):
                                isThere = True
                                cat = line2.split(',')[0]
                                sentiment = line2.split(',')[-1]
                                x = findIndex(cat)
                                vector[x] = vector[x] + 1.0
                                sentInd = checkSentiment(sentiment)
                                vector[sentInd] = vector[sentInd] + 1.0
                            break
            

    listOfVectors.append(vector)
    vector = [0.0] * categorysize
    
    
print listOfVectors[0]

 


# In[137]:

from itertools import chain
c=[]
myCollection = []

for x in listOfVectors:
    c = list(itertools.chain.from_iterable(zip(x,categoriesNew)))
    myCollection.append(c)
print myCollection[0]


# In[136]:

import numpy as np
import string
np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(threshold=np.inf)

normalizedVectors = []
a = np.empty((0))
indexBig = 0
indexSmall = 0
index = 0 
indexOne = 0
listOne = []
listTwo = []
NewListofLists = []
ListofLists = []

#normalize per whole list
# normalization
#categorysize = 2930
#number of tweets = 993
while index < categorysize:
    for x in listOfVectors:
        listOne.append(x[indexSmall])
    NewListofLists.append(listOne)
    listOne = []
    indexSmall = indexSmall + 1
    index = index + 1
        
for x in NewListofLists:
    x_np = np.asarray(x)
    np_minmax = (x_np - x_np.min()) / (x_np.max() - x_np.min())
    np_minmax = np_minmax.astype('float')
    np_minmax[np.isnan(np_minmax)] = 0
    normalizedVectors.append(np_minmax)        
        
ourLength = len(listOfVectors)

        
while indexOne < ourLength:
    for y in normalizedVectors:
        listTwo.append(y[indexBig])
    ListofLists.append(listTwo)
    listTwo = []
    indexOne = indexOne + 1
    indexBig = indexBig + 1
    
import sys
orig_stdout = sys.stdout
f = file('C:\Anaconda2\testing.txt', 'w')
sys.stdout = f

for zee in ListofLists:
    print zee, '\n'

sys.stdout = orig_stdout
f.close()



# In[ ]:



