#!/bin/sh

# export INST_NUM=2 or 3 or.. before running this

mkdir ~/dev
cd ~/dev/
git clone https://github.com/mkraemerx/sentiment-workshop.git

wget http://4530.hostserv.eu/resources/embed_tweets_de_100D_fasttext.zip
unzip embed_tweets_de_100D_fasttext.zip

source activate pytorch_p36
pip install torchtext
pip install --upgrade pip
conda install -y -c conda-forge spacy
python -m spacy download de

sudo locale-gen
sudo add-apt-repository -y ppa:certbot/certbot
sudo apt-get update
sudo apt-get install -y certbot

sudo certbot certonly -n --standalone --cert-name letsencrypt-cert --agree-tos --email michael.kraemer@innoq.com -d "c{$INST_NUM}.int.postlab.de"
sudo chgrp -R adm /etc/letsencrypt/archive
sudo chgrp -R adm /etc/letsencrypt/live
sudo chmod -R g+rx /etc/letsencrypt/archive
sudo chmod -R g+rx /etc/letsencrypt/live

cp sentiment-workshop/infra/start_jupyter.sh ~/
chmod +x ~/start_jupyter.sh

mkdir ~/.jupyter
cp sentiment-workshop/infra/jupyter_notebook_config.py ~/.jupyter/

sudo cp sentiment-workshop/infra/jupyter.service /etc/systemd/system/
sudo systemctl enable jupyter.service