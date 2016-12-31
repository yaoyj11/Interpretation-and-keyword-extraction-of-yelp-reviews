#!/bin/sh
#rm kwf/*
#
#for file in wt/*
#do
#  rm "$file"
#done
#rm wt/*
#
#rm html/*
#echo "run"
#python parsewordtext.py wordtext wt
#python parsekeywords.py keywords kwf html wordcloud/
#echo "stop"
scp -r html yjyao@login.cs.duke.edu:~/public_html/cs516
scp -r wt yjyao@login.cs.duke.edu:~/public_html/cs516

