import pickle as pkl
import numpy as np
import pandas as pd

# open the file to be read
with open('DB1processedEMG.pkl', 'rb') as f:
    emgData = pkl.load(f)
with open('DB1processedGlove.pkl', 'rb') as f:
    gloveData = pkl.load(f)
