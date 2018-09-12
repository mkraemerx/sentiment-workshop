#!/bin/sh

mkdir ~/dev
cd ~/dev/
git clone https://github.com/mkraemerx/sentiment-workshop.git

wget http://4530.hostserv.eu/resources/embed_tweets_de_100D_fasttext.zip
unzip embed_tweets_de_100D_fasttext.zip

source activate pytorch_p36
pip install torchtext
pip install --upgrade pip
conda install -c conda-forge spacy
python -m spacy download de

sudo add-apt-repository -y ppa:certbot/certbot
sudo apt-get update
sudo apt-get install -y certbot

sudo certbot certonly -n --standalone --cert-name letsencrypt-cert -d c1.int.postlab.de

cp sentiment-workshop/infra/start_jupyter.sh ~/
chmod +x ~/start_jupyter.sh

mkdir ~/.jupyter
cp sentiment-workshop/infra/jupyter_notebook_config.py ~/.jupyter/

sudo cp sentiment-workshop/infra/jupyter.service /etc/systemd/system/
sudo systemctl enable jupyter.service