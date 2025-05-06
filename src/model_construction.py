import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def build_model(input_dim):

   model = Sequential()
   model.add(Dense(256, activation='relu', input_shape=[input_dim]))
   model.add(Dropout(0.3)) 
   
   model.add(Dense(128, activation='relu'))
   model.add(Dropout(0.3)) 
   
   model.add(Dense(128, activation='relu'))
   model.add(Dropout(0.3)) 

   model.add(Dense(64, activation='relu'))

   model.add(Dense(32, activation='relu'))
   
   model.add(Dense(1))
   model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
   return model


def train_model(model, train_dataset, train_labels):
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        train_dataset, train_labels,
        epochs=500,
        validation_split=0.2,
        verbose=1,
        batch_size=32,
        callbacks=[early_stop]
    )
    return history

def model_plots(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist['rmse'] = np.sqrt(hist['mse'])
    hist['val_rmse'] = np.sqrt(hist['val_mse'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['mae'], name='mae', mode='markers+lines'))
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_mae'], name='val_mae', mode='markers+lines'))
    fig.update_layout(width=1000, height=500, title='MAE vs. VAL_MAE', xaxis_title='Epochs', yaxis_title='Mean Absolute Error', yaxis_type='log')
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['rmse'], name='rmse', mode='markers+lines'))
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_rmse'], name='val_rmse', mode='markers+lines'))
    fig.update_layout(width=1000, height=500, title='RMSE vs. VAL_RMSE', xaxis_title='Epochs', yaxis_title='Root Mean Squared Error', yaxis_type='log')
    fig.show()
