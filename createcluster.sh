#!/bin/sh
/home/yaoyj11/Downloads/spark-1.6*/ec2/spark-ec2 --key-pair=yaoyj11ec2 --identity-file=/home/yaoyj11/.ssh/yaoyj11ec2.pem --region=us-east-1 --instance-type=t2.large --slaves=10 launch sparkcluster
