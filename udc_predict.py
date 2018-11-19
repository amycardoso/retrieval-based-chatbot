import os
import time
import itertools
import sys
import numpy as np
import tensorflow as tf
import udc_model
import udc_hparams
import udc_metrics
import udc_inputs
import pandas as pd
from models.dual_encoder import dual_encoder_model
from models.helpers import load_vocab

tf.flags.DEFINE_string("model_dir", None, "Directory to load model checkpoints from")
tf.flags.DEFINE_string("vocab_processor_file", "./data/vocab_processor.bin", "Saved vocabulary processor file")
FLAGS = tf.flags.FLAGS

if not FLAGS.model_dir:
  print("You must specify a model directory")
  sys.exit(1)

def tokenizer_fn(iterator):
  return (x.split(" ") for x in iterator)

# Load vocabulary
vp = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(
  FLAGS.vocab_processor_file)

# Load your own data here
#INPUT_CONTEXT = "anyon know whi my stock oneir export env var usernam ' ? i mean what be that use for ? i know of $ user but not $ usernam . my precis instal doe n't export usernam __eou__ __eot__ look like it use to be export by lightdm , but the line have the comment `` // fixm : be this requir ? '' so i guess it be n't surpris it be go __eou__ __eot__ thank ! how the heck do you figur that out ? __eou__ __eot__ https : //bugs.launchpad.net/lightdm/+bug/864109/comments/3 __eou__ __eot__"
#POTENTIAL_RESPONSES = ["nice thank ! __eou__", "everi time the kernel chang , you will lose video __eou__ yep __eou__", "ok __eou__", "! nomodeset > acer __eou__ i 'm assum it be a driver issu . __eou__ ! pm > acer __eou__ i do n't pm . ; ) __eou__ oop sorri for the cap __eou__", "http : //www.ubuntu.com/project/about-ubuntu/deriv ( some call them deriv , other call them flavor , same differ ) __eou__", "thx __eou__ unfortun the program be n't instal from the repositori __eou__", "how can i check ? by do a recoveri for test ? __eou__", "my humbl apolog __eou__", "# ubuntu-offtop __eou__"]

# Load your own data here
#INPUT_CONTEXT = "Example context"
#POTENTIAL_RESPONSES = ["Response 1", "Response 2"]

test_df = pd.read_csv("./data/test.csv")
elementId = 2
INPUT_CONTEXT = test_df.Context[elementId]
POTENTIAL_RESPONSES = test_df.iloc[elementId,1:].values

def get_features(context, utterance):
  context_matrix = np.array(list(vp.transform([context])))
  utterance_matrix = np.array(list(vp.transform([utterance])))
  context_len = len(context.split(" "))
  utterance_len = len(utterance.split(" "))
  features = {
    "context": tf.convert_to_tensor(context_matrix, dtype=tf.int64),
    "context_len": tf.constant(context_len, shape=[1,1], dtype=tf.int64),
    "utterance": tf.convert_to_tensor(utterance_matrix, dtype=tf.int64),
    "utterance_len": tf.constant(utterance_len, shape=[1,1], dtype=tf.int64),
  }
  return features, None

if __name__ == "__main__":
  hparams = udc_hparams.create_hparams()
  model_fn = udc_model.create_model_fn(hparams, model_impl=dual_encoder_model)

  estimator = tf.contrib.learn.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir)

  # Ugly hack, seems to be a bug in Tensorflow
  # estimator.predict doesn't work without this line
  #estimator._targets_info = tf.contrib.learn.estimators.tensor_signature.TensorSignature(tf.constant(0, shape=[1,1]))

  for r in POTENTIAL_RESPONSES:
    prob = estimator.predict(input_fn=lambda: get_features(INPUT_CONTEXT, r))
    #print("{}: {:g}".format(r, prob[0,0])) 	    
    for k in list(prob):
      print(r + ", " + str(k[0]))
  print("Done.") 
