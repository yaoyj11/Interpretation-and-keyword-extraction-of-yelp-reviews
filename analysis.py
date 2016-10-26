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
    stars=tup[0]
    text=tup[1]
    words=re.split(r"\s+",remove_punctuation(text))
    nonEmpty=[elem for elem in words if elem!=""]
    s=""
    for w in nonEmpty:
        s=s+w+" "
    x=array('d')
    for f in features:
        x.append(s.count(f))
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
            .flatMap(parseWord)\
            .map(lambda x: (x,1))\
            .reduceByKey(lambda x,y: x+y)
    print("training set size: "+str(train_set.count()))
    positive=stars_wordcount.filter(lambda x:x[0][0]==1)
    negative=stars_wordcount.filter(lambda x:x[0][0]==0)

    pfeatures=positive.takeOrdered(1000,key=lambda x:-x[1])
    nfeatures=negative.takeOrdered(1000,key=lambda x:-x[1])
    features=set()
    for f in pfeatures:
        features.add(f[0][1])
    for f in nfeatures:
        features.add(f[0][1])
    features=list(features)
    print(features)
    print(str(len(features))+" features get")

    #parse data as labeled vectors
    train_data=train_set.map(lambda x:mapLabeled(x,features))
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
        coeff.append((features[i],model._coeff[i]))
    #coeffrdd= sc.parallelize(coeff).foreach(print)
    coeffrdd= sc.parallelize(coeff)
    top=coeffrdd.takeOrdered(100,lambda x: -math.fabs(x[1]))
    for e in top:
        print(e)
    coeffrdd.saveAsTextFile(modeldir)
    


