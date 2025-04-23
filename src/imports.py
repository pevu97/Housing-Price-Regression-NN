def all_imports():

  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns
  import plotly.express as px
  import plotly.graph_objects as go
  import tensorflow as tf
  
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense, Dropout
  from tensorflow.keras.callbacks import EarlyStopping
  
  !git clone https://github.com/pevu97/Housing-Price-Regression-NN.git
  
  
  np.set_printoptions(precision=3, suppress=True, linewidth=150)
  pd.options.display.float_format = '{:.6f}'.format
  tf.__version__
  raw_dataset = pd.read_csv('Housing-Price-Regression-NN/data/housing.csv')

