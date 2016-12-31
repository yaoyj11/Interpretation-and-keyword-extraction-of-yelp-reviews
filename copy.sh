#!/bin/sh
scp -i ~/.ssh/yaoyj11ec2.pem -r *.py run.sh  *.sh *.zip ec2-user@$SPARK_MASTER:~/
