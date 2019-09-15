
## Analysis Of The Context Size Impact In Deep Learning Conversational Systems
This repository holds an implementation of the Deep Learning Dual Encoder LSTM Model and the Vector Space Model, both used to evaluate and analyze the impact of the context size over the quality of responses.
[![chatbot](https://blog-assets.freshworks.com/freshdesk/wp-content/uploads/2018/08/Header_gif_assembly-1.gif "chatbot")](https://blog-assets.freshworks.com/freshdesk/wp-content/uploads/2018/08/Header_gif_assembly-1.gif "chatbot")
### Overview
Some of the codes used here were produced in this [hands-on](http://www.wildml.com/2016/07/deep-learning-for-chatbots-2-retrieval-based-model-tensorflow "hands-on"), which implement the Dual Encoder LSTM Model, from this [paper](http://arxiv.org/abs/1506.08909 "paper"), also implement the Vector Space Model, which in this research was used as a baseline.

### Configuration

The codes use Python 2 and 3. Clone the repository and install all necessary packages:
> Note:  Only database generation scripts use Python2.
```
1. install tensorflow (version 0.11 and above work correctly, version 0.10 not tested)
2. (optional) install cuda + cudnn (recommend for gpu support)
2. pip3 install -U pip
3. pip3 install -r requirements.txt
```

### Dialogue dataset


Experiments can be performed using Ubuntu Dialogue Corpus version 2.0 featured in this [paper](http://www.cs.toronto.edu/~lcharlin/papers/ubuntu_dialogue_dd17.pdf "paper"), whose generation script is available in this [repository](https://github.com/rkadlec/ubuntu-ranking-dataset-creator "repository"). However, since the goal of the research was to understand the impact of context size on predicting the next utterance, it was necessary to modify the generation script to get training sets with the number of turns informed by argument. Thus, the modified script can be found in the scripts [folder](https://github.com/amycardoso/retrieval-based-chatbot/blob/master/scripts/create_ubuntu_dataset_modificado.py "folder").

#### Modified script training set generation
For the generation of training sets, follow the steps described in this [repository](https://github.com/rkadlec/ubuntu-ranking-dataset-creator "repository"), except for the addition of a sub parser to the training parser to determine the desired number of turns.

#### Subparser:

`train`: training set generator

-   `-t`: desired number of turns

Example for generating a set consisting of contexts with 2 turns:
```
python2 create_ubuntu_dataset_modificado.py --data_root ./dados -o 'train.csv' -t -s -l train -t 2
```
Run training set generation with the modified script, but for validation and test sets use the original script, or download all required sets [here](https://drive.google.com/open?id=1-1LbkFMUIx6J3hqHFMrVtdPTkp5K9FY "here"). Finally, move all files to the `./Data` folder.

#### Preprocessing
Before moving to Deep Learning model training, sets need to be transformed from CSV to TFRecord.
```
cd scripts
python3 prepare-data.py
```
### Dual Encoder LSTM Model
#### Training

```
python3 udc_train.py
```

#### Evaluation

```
python3 udc_test.py --model_dir=...
```


**Example:**
```
python3 udc_test.py --model_dir=./runs/1481183770/
```

#### Prediction

```
python3 udc_predict.py --model_dir=...
```

**Example:**
```
python3 udc_predict.py --model_dir=./runs/1481183770/
```

### Vector Space Model

As a baseline, we used the Vector Space Model, available in the [notebooks](https://github.com/amycardoso/retrieval-based-chatbot/tree/master/notebooks "notebooks") folder.
