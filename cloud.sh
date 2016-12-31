#!/bin/bash
for file in kwf/*
do
  filename=${file##*/}
  wordcloud_cli.py --text "kwf/$filename" --imagefile "wordcloud/$filename.png" --freq True
done
