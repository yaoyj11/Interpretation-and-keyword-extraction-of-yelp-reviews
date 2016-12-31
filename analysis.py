from __future__ import print_function
from operator import add
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from parser import *
import re
import random
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
import numpy
from array import array
import math
#test



def tryprint(text):
    try:
        print(text)
    except:
        print("print error")
    


def parseMultiWordNLTK((stars,text),types):
    res=[]
    import nltk
    words=nltk.word_tokenize(text)
    wtags=nltk.pos_tag(words)
    #tryprint(text)
    lastw=""
    lastt=""
    for elem in wtags:
        if(elem[1] in types):
            res.append(elem[0].lower())
            if(lastw!="" and (lastt+elem[1])in types):
                res.append(lastw+" "+elem[0].lower())
            lastw=elem[0].lower()
            lastt=elem[1]
    good=0
    if(stars>3):
        good=1
    return (good,res)


def parseStarsText(line):
    star=int(line[29:30])
    text=line[34:-3]
    bid=line[3:25]
    #tryprint(line)
    #tryprint(text)
    return (bid,(star,text))

def parseModel(line):
    line=line[2:-1]
    s=line.split(", ")
    word=s[0][1:-1]
    coeff=float(s[1])
    score=float(s[2])
    return (word,coeff,score)

def mapLabeled(tup,features):
    import nltk
    stars=tup[0]
    text=tup[1]
    words=re.split(r"\s+",remove_punctuation(text))
    nonEmpty=[elem for elem in words if elem!=""]
    s=""
    for w in nonEmpty:
        s=s+w+" "
    x=array('d')
    for f in features:
        #x.append(s.count(f))
        if(s.count(f)>0):
            x.append(1)
        else:
            x.append(0)
    good=0
    if stars>3:
        good=1
    return LabeledPoint(good,x)

def mapLabeledNLTK(tups,features,types):
    x=[]
    for f in features:
        #x.append(s.count(f))
        if(f in tups[1]):
            x.append(1)
        else:
            x.append(0)
    return LabeledPoint(tups[0],x)

def mapSentences(tup):
    bid=tup[0]
    text=tup[1][1]
    ss=re.split(r"[,\.]",text)
    res=[]
    for s in ss:
        if(remove_punctuation(s).replace(" ","")!=""):
            res.append((bid,s))
    return res

def getKeyWords(tup,coeff,types=[]):
    bid=tup[0]
    #print(tup[1])
    text=tup[1]
    import nltk
    words=nltk.word_tokenize(text)
    wtags=nltk.pos_tag(words)
    #tryprint(text)
    lastw=""
    allwords=[]
    for elem in wtags:
        if elem[1] in types:
            allwords.append(elem[0])
            if(lastw!=""):
                allwords.append(lastw+" "+elem[0])
            lastw=elem[0]
    coeff_of_sentence=0
    for w in allwords:
        if w in coeff:
            coeff_of_sentence+=coeff[w]
    res=[]
    for elem in wtags:
        if elem[1] in ["NN","NNP","NNPS","NNS"]:
            res.append(((bid,elem[0]),coeff_of_sentence))
    return res

#def writeFeatures(feature):
#    f=open("features.txt","w")
#    for line in f:
#        featurefile.write(line+"\n")
#    f.close()
#
#def readFeatures(featurefile):
#    f=open("features.txt")
#    feature=[]
#    for line in f:
#        feature.append(line.replace("\n","")
#    f.close()
#    return feature

def readFromDataset(sc,inputdir,reviewfile,businessfile,outputdir,trainpath,validationpath,testpath,num_partition=10):
    print("read reviews\n")
    reviews=sc.textFile(inputdir+reviewfile,num_partitions).map(parseReview)
    #(business_id,[categories])
    print("read restaurants\n")
    restaurants=sc.textFile(inputdir+businessfile,num_partitions)\
            .map(parseBusiness)\
            .filter(lambda x:"Restaurants" in  x[1])

    print("filter reviews for restaurant")

    stars_text=reviews.join(restaurants)\
            .map(lambda x:(x[0],x[1][0]))\
            .map(lambda x: (x,random.randint(0,9)))

    train_set = stars_text.filter(lambda x:x[1]<=7)\
            .map(lambda x:x[0])
    train_set.saveAsTextFile(outputdir+trainpath)
    validation_set=stars_text.filter(lambda x:x[1]==8)\
            .map(lambda x:x[0])
    validation_set.saveAsTextFile(outputdir+validationpath)

    test_set=stars_text.filter(lambda x:x[1]==9)\
            .map(lambda x:x[0])
    test_set.saveAsTextFile(outputdir+testpath)
    return train_set,validation_set,test_set

def readProcessedData(sc,outputdir,trainpath,validationpath,testpath):
    train_set=sc.textFile(outputdir+trainpath+"/*").map(parseStarsText)
    validation_set=sc.textFile(outputdir+validationpath+"/*").map(parseStarsText)
    test_set=sc.textFile(outputdir+testpath+"/*").map(parseStarsText)
    return train_set,validation_set,test_set

def uniqueFeatures(pf,nf,count=100):
    pfeatures=[elem[0][1] for elem in pf]
    nfeatures=[elem[0][1] for elem in nf]
    uniquepf=[]
    uniquenf=[]
    for f in pfeatures:
        if f not in nfeatures:
            uniquepf.append(f)
    for f in nfeatures:
        if f not in pfeatures:
            uniquenf.append(f)
    print("unique positive features")
    print(uniquepf[0:count])
    print("negative features")
    print(uniquenf[0:count])
    return uniquepf[0:count],uniquenf[0:count]

#input (bid,[(word,score)])
def topN(tup,N):
    bid=tup[0]
    words=[x for x in tup[1]]
    words=sorted(words,key=lambda x:x[1])
    res=[]
    if(len(words)>2*N):
        res=(words[0:N],words[len(words)-N:-1])
    else:
        res=(words,words)
    return (bid,res)

def mapFeatures(line):

    #s=line[3:-1].split("', ")
    s=re.split(r"['\"],\s",line[3:-1])
    #print(line)
    #print(s)
    feature=s[0]
    occ=int(s[1])
    return (feature,occ)

if __name__ == "__main__":
    conf = SparkConf()
    conf.setMaster("local[8]").setAppName("YELP")
    sc = SparkContext(conf=conf)
    log4j = sc._jvm.org.apache.log4j
    log4j.LogManager.getRootLogger().setLevel(log4j.Level.ERROR)
    print("Set log level to Error")
    num_partitions=10
    inputdir="yelp_dataset/"
    reviewfile="yelp_academic_dataset_review.json"
    businessfile="yelp_academic_dataset_business.json"
    outputdir="output/"
    featuredir=outputdir+"feature"
    pfeaturedir=outputdir+"pfeature"
    nfeaturedir=outputdir+"nfeature"
    modeldir=outputdir+"model"
    kwdir=outputdir+"kw"
    trainpath="train"
    validationpath="validate"
    testpath="test"
    types=["CC","CD","DT","JJ","JJR","JJS","MD","NN","RB","RBR","RBS","VB","VBD","VBG","VBN","VBP","VBZ","RBVB","RBJJ"]
    N=15

    # Initialize the spark context.

    #(business_id,(stars,text))
    #(train_set,validation_set,test_set)=readFromDataset(sc,inputdir,reviewfile,businessfile,outputdir,trainpath,validationpath,testpath,num_partitions)
    (train_set,validation_set,test_set)=readProcessedData(sc,outputdir,trainpath,validationpath,testpath)

#    (good,wordslist)
    train_set=train_set.union(test_set)
    stars_words=train_set\
            .map(lambda x: x[1])\
            .map(lambda x:parseMultiWordNLTK(x,types))
    #((good,word),count)
    stars_wordcount=stars_words\
            .flatMap(lambda x:[(x[0],e) for e in x[1]])\
            .map(lambda x: (x,1))\
            .reduceByKey(lambda x,y: x+y)
    print("training set size: "+str(train_set.count()))
    positive=stars_wordcount.filter(lambda x:x[0][0]==1)
    negative=stars_wordcount.filter(lambda x:x[0][0]==0)

    pfeatures=positive.takeOrdered(2000,key=lambda x:-x[1])
    sc.parallelize(pfeatures).saveAsTextFile(pfeaturedir)
    nfeatures=negative.takeOrdered(2000,key=lambda x:-x[1])
    sc.parallelize(nfeatures).saveAsTextFile(nfeaturedir)
    #####
    #Task 1: Most frequently used words in positive an dnegative reviews, remove common ones
    #(stars,word),count
    #####
    (uniquepf,uniquenf)=uniqueFeatures(pfeatures,nfeatures)

    #####
    #Task 2: Most frequently used words in positive an dnegative reviews as features, keep common ones
    #####
    #features=set()
    #for f in pfeatures:
    #    features.add(f[0][1])
    #for f in nfeatures:
    #    features.add(f[0][1])
    #features=list(features)

    #take unique features
    features=uniquepf+uniquenf
    print(features)
    print(str(len(features))+" features get")
    occurencies={}
    #features=[e[0][1] for e in features]
    for f in features:
        c1=0
        c2=0
        for p in pfeatures:
            if(p[0][1]==f):
                c1=p[1]
                break
        for p in nfeatures:
            if(p[0][1]==f):
                c2=p[1]
                break
        occurencies[f]=c1+c2
    #save features##
    featuresrdd=sc.parallelize([(k,v) for k,v in occurencies.items()])
    featuresrdd.saveAsTextFile(featuredir)
    #read features#
    features_occurencies=sc.textFile(featuredir+"/*").map(mapFeatures).collect()
    features=[f[0] for f in features_occurencies]
    occurencies={}
    for f in features_occurencies:
        occurencies[f[0]]=f[1]

    #parse data as labeled vectors
    #(good,words)=>(good,vector)
    train_data=stars_words\
            .map(lambda x:mapLabeledNLTK(x,features,types))
    validation_data=validation_set\
            .map(lambda x: x[1])\
            .map(lambda x:parseMultiWordNLTK(x,types))\
            .map(lambda x:mapLabeledNLTK(x,features,types))
    #train
    print("training...")
    model = LogisticRegressionWithSGD.train(train_data)
    coeff=[]
    for i in range(len(features)):
        coeff.append((features[i],model._coeff[i],occurencies[features[i]]*model._coeff[i]))
    print("model intercept:")
    print(str(model._intercept))

    #coeffrdd= sc.parallelize(coeff).foreach(print)
    coeffrdd= sc.parallelize(coeff)
    topcoeff=coeffrdd.takeOrdered(100,lambda x: -math.fabs(x[1]))
    print("features with highest cofficient")
    for e in topcoeff:
        print(e)
    topinfluence=coeffrdd.takeOrdered(100,lambda x: -math.fabs(x[2]))
    print("features with highest influence")
    for e in topinfluence:
        print(e)
    coeffrdd.saveAsTextFile(modeldir)
    #train_err
    train_label_preds=train_data.map(lambda point: (point.label,model.predict(point.features)))
    trainErr=train_label_preds.filter(lambda (v,p): v!=p).count()/float(train_data.count())
    print("training finished, training error: "+str(trainErr))

    #valid_err
    valid_label_preds=validation_data.map(lambda point:(point.label,model.predict(point.features)))
    validErr=valid_label_preds.filter(lambda (v,p):v!=p).count()/float(validation_data.count())
    TP=valid_label_preds.filter(lambda (v,p): v==1 and p ==1).count()
    TN=valid_label_preds.filter(lambda (v,p): v==0 and p ==0).count()
    FP=valid_label_preds.filter(lambda (v,p): v==0 and p ==1).count()
    FN=valid_label_preds.filter(lambda (v,p): v==1 and p ==0).count()

    try:
        print("validation error"+str(validErr))
        print("precision "+str(TP/float(TP+FP)))
        print("recall "+str(TP/float(TP+FN)))
        print("F-measure "+str(2*(TP/float(TP+FP))*(TP/float(TP+FN))/(TP/float(TP+FP)+TP/float(TP+FN))))
    except:
        print("error")

    #####
    #Task 3: Find representative words for each restaurant.
    #####
    #read model
#    coeffrdd=sc.textFile(modeldir+"/*").map(parseModel)
#    coeff=coeffrdd.collect()
#    #{feature:coeff}
#    coeffmap={}
#    for c in coeff:
#        coeffmap[c[0]]=c[1]
#    dataset=train_set.union(validation_set).union(test_set)
#    #(bid,(stars,text))=>(bid,sentence)=>((bid,wordnoun),score)=>(bid,[(noun,score)])
#    keywords=dataset\
#        .flatMap(mapSentences)\
#        .flatMap(lambda x:getKeyWords(x,coeffmap,types))\
#        .reduceByKey(lambda x,y:x+y)\
#        .filter(lambda x: x[1]>5 or x[1]<-5)
#        .map(lambda x:(x[0][0],(x[0][1],x[1])))\
#        .groupByKey()\
#        .map(lambda x:topN(x,N))
##        .saveAsTextFile(kwdir)
#    #    .foreach(print)
#    keywords.saveAsTextFile(kwdir)
#
#    keywords.foreach(print)
#
