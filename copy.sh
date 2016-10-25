#!/bin/sh
scp -i ~/.ssh/yaoyj11ec2.pem -r *.py  ec2-user@$SPARK_MASTER:~/
