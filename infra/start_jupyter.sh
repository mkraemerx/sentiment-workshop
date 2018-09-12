#!/bin/bash

export PATH=~/anaconda3/bin:$PATH
source activate pytorch_p36

echo "PATH: $PATH"
JUP_ENV=`conda env list | grep "*" | awk '{print $1}'`
echo "starting jupyter in conda env $JUP_ENV"

cd /home/ubuntu/dev
/home/ubuntu/anaconda3/envs/pytorch_p36/bin/jupyter-notebook --config=/home/ubuntu/.jupyter/jupyter_notebook_config.py
