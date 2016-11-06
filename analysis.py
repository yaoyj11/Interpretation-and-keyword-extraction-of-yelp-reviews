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

def parseWord((stars,text)):
    res=[]
    line=remove_punctuation(text)

    words=re.split(r"\s+",line)
    nonEmpty=[elem for elem in words if elem!=""]

    for w in nonEmpty:
        if(stars>3):
            res.append((1,w))
        else:
            res.append((0,w))
    return res

def tryprint(text):
    try:
        print(text)
    except:
        print("print error")
    

def parseWordNLTK((stars,text)):
    import nltk
    res=[]
    words=nltk.word_tokenize(text)
    wtags=nltk.pos_tag(words)
    tryprint(text)
    types=["CC","CD","CT","JJ","JJR","JJS","MD","NN","NNP","NNPS","NNS","PDT","RB","RBR","RBS","VB","VBD","VBG","VBN","VBP","VBZ"]
    for elem in wtags:
        if(elem[1] in types):
            if(stars>3):
                res.append((1,elem[0]))
            else:
                res.append((0,elem[0]))
    return res

def parseMultiWordNLTK((stars,text)):
    res=[]
    import nltk
    words=nltk.word_tokenize(text)
    wtags=nltk.pos_tag(words)
    #tryprint(text)
    types=["DT","JJ","JJR","JJS","MD","NN","NNP","NNPS","NNS","PDT","RB","RBR","RBS","VB","VBD","VBG","VBN","VBP","VBZ"]
    lastw=""
    for elem in wtags:
        if(elem[1] in types):
            if(stars>3):
                res.append((1,elem[0]))
            else:
                res.append((0,elem[0]))
            if(lastw!=""):
                if(stars>3):
                    res.append((1,lastw+" "+elem[0]))
                else:
                    res.append((0,lastw+" "+elem[0]))
            lastw=elem[0]
    return res


def parseMultiWord((stars,text)):
    res=[]
    line=remove_punctuation(text)

    words=re.split(r"\s+",line)
    nonEmpty=[elem for elem in words if elem!=""]
    for w in nonEmpty:
        if(stars>3):
            res.append((1,w))
        else:
            res.append((0,w))
    for i in range(len(nonEmpty)-1):
        if(stars>3):
            res.append((1,nonEmpty[i]+" "+nonEmpty[i+1]))
        else:
            res.append((0,nonEmpty[i]+" "+nonEmpty[i+1]))

    return res

def parseStarsText(line):
    star=int(line[1:2])
    text=line[6:-2]
    print(line)
    print(text)
    return (star,text)

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

def mapLabeledNLTK(tup,features):
    stars=tup[0]
    text=tup[1]
    import nltk
    words=nltk.word_tokenize(text)
    wtags=nltk.pos_tag(words)
    #tryprint(text)
    types=["DT","JJ","JJR","JJS","MD","NN","NNP","NNPS","NNS","PDT","RB","RBR","RBS","VB","VBD","VBG","VBN","VBP","VBZ"]
    lastw=""
    l=[]
    for elem in wtags:
        if(elem[1] in types):
            l.append(elem[0])
            if(lastw!=""):
                l.append(lastw+" "+elem[0])
            lastw=elem[0]
    x=[]
    for f in features:
        #x.append(s.count(f))
        if(f in l):
            x.append(1)
        else:
            x.append(0)
    good=0
    if stars>3:
        good=1
    return LabeledPoint(good,x)

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
            .map(lambda x:x[1][0])\
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

if __name__ == "__main__":
    num_partitions=10
    inputdir="yelp_dataset/"
    reviewfile="yelp_review_part.json"
    businessfile="yelp_business_part.json"
    outputdir="output/"
    modeldir=outputdir+"model"
    trainpath="train"
    validationpath="validate"
    testpath="test"
    conf = SparkConf()
    conf.setMaster("local").setAppName("YELP")
    sc = SparkContext(conf=conf)
    log4j = sc._jvm.org.apache.log4j
    log4j.LogManager.getRootLogger().setLevel(log4j.Level.ERROR)
    print("Set log level to Error")

    # Initialize the spark context.

    #(business_id,(stars,text))
    (train_set,validation_set,test_set)=readFromDataset(sc,inputdir,reviewfile,businessfile,outputdir,trainpath,validationpath,testpath,num_partitions)
    #(train_set,validation_set,test_set)=readProcessedData(sc,outputdir,trainpath,validationpath,testpath)

    stars_wordcount=train_set\
            .flatMap(parseMultiWordNLTK)\
            .map(lambda x: (x,1))\
            .reduceByKey(lambda x,y: x+y)
    print("training set size: "+str(train_set.count()))
    positive=stars_wordcount.filter(lambda x:x[0][0]==1)
    negative=stars_wordcount.filter(lambda x:x[0][0]==0)

    pfeatures=positive.takeOrdered(1000,key=lambda x:-x[1])
    nfeatures=negative.takeOrdered(1000,key=lambda x:-x[1])
    (uniquepf,uniquenf)=uniqueFeatures(pfeatures,nfeatures)
    features=set()
    for f in pfeatures:
        features.add(f[0][1])
    for f in nfeatures:
        features.add(f[0][1])
    features=list(features)
    print(features)
    print(str(len(features))+" features get")
    occurencies={}
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

    #parse data as labeled vectors
    train_data=train_set.map(lambda x:mapLabeledNLTK(x,features))
    validation_data=validation_set.map(lambda x:mapLabeled(x,features))
    #train
    print("training...")
    model = LogisticRegressionWithSGD.train(train_data)
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

    print("validation error"+str(validErr))
    print("precision "+str(TP/float(TP+FP)))
    print("recall "+str(TP/float(TP+FN)))
    print("F-measure "+str(2*(TP/float(TP+FP))*(TP/float(TP+FN))/(TP/float(TP+FP)+TP/float(TP+FN))))
    coeff=[]
    for i in range(len(features)):
        coeff.append((features[i],model._coeff[i],occurencies[features[i]]*model._coeff[i]))
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
