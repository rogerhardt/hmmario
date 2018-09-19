#!/usr/bin/env python
import sys, json, codecs, pickle, argparse
import numpy as np

from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from hmmlearn import hmm
from nltk import FreqDist
from datetime import datetime

def warn(msg):
    print(msg, file=sys.stderr)
	
def outfile(seed, ext):
    return "hmmario.{seed}.{n}.{ext}".format(
            seed=seed, n=args.num_states, ext=ext)		
			
def initiateModel():
    model = hmm.MultinomialHMM(n_components=args.num_states, n_iter=55, init_params='ste', verbose='true')
    return model

#np.random.seed(seed=None)

args = argparse.ArgumentParser(
        description="Train HMM with Mario Level Input")
args.add_argument("-n", "--num-states", type=int, required=True,
        help="Mumber of hidden states")
args.add_argument("-s", "--num-samples", type=int, required=True,
        help="Mumber of samples to generate")
args.add_argument("input", nargs="?", type=argparse.FileType("r"),
        default=sys.stdin,
        help="Training input of mario levels, horizontaly transposed")
args = args.parse_args()

print (args.input)

lines = [line.split() for line in args.input]
words = [word.lower() for line in lines for word in line]

alphabet = set(words)

le = LabelEncoder()
le.fit(list(alphabet))

seq = le.transform(words)
features = np.fromiter(seq, np.int64)
features = np.atleast_2d(features).T

model = initiateModel()

lengths = [len(line) for line in lines]

model = model.fit(features)

milis = datetime.now().microsecond

joblib.dump(model, outfile("mdl",milis))
with open(outfile("le",milis), "wb") as f:
    pickle.dump(le, f)

warn("Trained model written for backup to:\n\t- {0}\n\t- {1}".format(
        outfile("mdl", milis), outfile("le", milis)
    ))
	
for _i in range(args.num_samples):
   #generates a level 100 blocks wide
   symbols, states = model.sample(100)
   
   sample = le.inverse_transform(np.squeeze(symbols))
   
   #prints the level transposed to become human readable
   formattedSample = ([''.join(s) for s in zip(*sample)])
   for block in formattedSample:
   	print (block)
   print()