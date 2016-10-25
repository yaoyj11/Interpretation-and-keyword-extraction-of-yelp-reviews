from __future__ import print_function
from operator import add
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from parser import *
import re

def parseWord((stars,text)):
    res=[]
    line=remove_punctuation(text)

    words=re.split(r"\s+",line)

    for w in words:
        if(w!=""):
            if(stars>3):
                res.append((1,w))
            else:
                res.append((0,w))
    return res

    
if __name__ == "__main__":
    num_partitions=10
    inputdir="yelp_dataset/"
    reviewfile="yelp_review_part.json"
    businessfile="yelp_business_part.json"
    conf = SparkConf()
    conf.setMaster("local").setAppName("YELP")
    sc = SparkContext(conf=conf)
    log4j = sc._jvm.org.apache.log4j
    log4j.LogManager.getRootLogger().setLevel(log4j.Level.ERROR)

    # Initialize the spark context.

    # Loads in input file. It should be in format of:
    #     URL         neighbor URL
    #     URL         neighbor URL
    #     URL         neighbor URL
    #     ...
    #(business_id,(stars,text))
    print("read reviews\n")
    reviews=sc.textFile(inputdir+reviewfile,num_partitions).map(parseReview)
    #(business_id,[categories])
    print("read restaurants\n")
    restaurants=sc.textFile(inputdir+businessfile,num_partitions)\
            .map(parseBusiness)\
            .filter(lambda x:"Restaurants" in  x[1])
    print("filter reviews for restaurant")
    stars_wordcount=reviews.join(restaurants)\
            .map(lambda x:x[1][0])\
            .flatMap(parseWord)\
            .map(lambda x: (x,1))\
            .reduceByKey(lambda x,y: x+y)
    positive=stars_wordcount.filter(lambda x:x[0][0]==1)\
            .takeOrdered(50,lambda x:-x[1])
    print("positive")
    for p in positive:
        print(p)
    print("negative")
    negative=stars_wordcount.filter(lambda x:x[0][0]==0)\
            .takeOrdered(50,lambda x:-x[1])
    for p in negative:
        print(p)





