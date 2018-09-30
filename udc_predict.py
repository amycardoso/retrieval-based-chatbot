import time
import sys
import numpy as np
import tensorflow as tf
import udc_model
import udc_hparams
from models.dual_encoder import dual_encoder_model
import pandas as pd
from termcolor import colored

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

# Load data for predict
test_df = pd.read_csv("./data/test.csv")
elementId = 0
INPUT_CONTEXT = test_df.Context[elementId]
POTENTIAL_RESPONSES = test_df.iloc[elementId,1:].values

def get_features(context, utterances):
  context_matrix = np.array(list(vp.transform([context])))
  utterance_matrix = np.array(list(vp.transform([utterances[0]])))
  context_len = len(context.split(" "))
  utterance_len = len(utterances[0].split(" "))
  features =  {
        "context": tf.convert_to_tensor(context_matrix, dtype=tf.int64),
        "context_len": tf.constant(context_len, shape=[1,1], dtype=tf.int64),
        "utterance": tf.convert_to_tensor(utterance_matrix, dtype=tf.int64),
        "utterance_len": tf.constant(utterance_len, shape=[1,1], dtype=tf.int64),
        "len":len(utterances)
  }

  for i in range(1,len(utterances)):
      utterance = utterances[i];

      utterance_matrix = np.array(list(vp.transform([utterance])))
      utterance_len = len(utterance.split(" "))

      features["utterance_{}".format(i)] = tf.convert_to_tensor(utterance_matrix, dtype=tf.int64)
      features["utterance_{}_len".format(i)] = tf.constant(utterance_len, shape=[1,1], dtype=tf.int64)

  return features, None

if __name__ == "__main__":
  # tf.logging.set_verbosity(tf.logging.INFO)
  hparams = udc_hparams.create_hparams()
  model_fn = udc_model.create_model_fn(hparams, model_impl=dual_encoder_model)

  estimator = tf.contrib.learn.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir)

  starttime = time.time()

  if float(tf.__version__[0:4])<0.12: #check TF version to select method
      prob = estimator.predict(input_fn=lambda: get_features(INPUT_CONTEXT, POTENTIAL_RESPONSES),as_iterable=True)
  else:
      prob = estimator.predict(input_fn=lambda: get_features(INPUT_CONTEXT, POTENTIAL_RESPONSES))
  results = next(prob)

  endtime = time.time()

  print('\n')
  print(colored('[Predict time]', on_color='on_blue',color="white"),"%.2f sec" % round(endtime - starttime,2))
  print(colored('[     Context]', on_color='on_blue',color="white"),INPUT_CONTEXT)
  # print("[Results value ]",results)
  answerId = results.argmax(axis=0)
  if answerId==0:
      print(colored('[      Answer]', on_color='on_green'), POTENTIAL_RESPONSES[answerId])
  else:
      print (colored('[      Answer]', on_color='on_red'),POTENTIAL_RESPONSES[answerId])
      print (colored('[Right answer]', on_color='on_green'), POTENTIAL_RESPONSES[0])
