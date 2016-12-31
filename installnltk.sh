#!/bin/bash
#setup tools
sudo yum update
sudo yum install python27
curl https://bitbucket.org/pypa/setuptools/raw/bootstrap/ez_setup.py | sudo /usr/bin/python27
sudo easy_install pip
echo "alias python='python27'" >> ~/.bashrc
source ~/.bashrc


