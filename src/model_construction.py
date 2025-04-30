def build_model():
  model = Sequential()
  model.add(Dense(1024, input_shape=[len(train_dataset_scaled.columns)], activation='relu', kernel_regularizer='l2'))
  model.add(Dropout(0.2))
  model.add(Dense(512, activation='relu', kernel_regularizer='l2'))
  model.add(Dense(128, activation='relu', kernel_regularizer='l2'))
  model.add(Dense(1))

  model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
  return model

  def train_model():
    from tensorflow.keras.callbacks import EarlyStopping

    early_stop = EarlyStopping(
    monitor='val_loss',       # na jakiej metryce się skupiamy
    patience=5,               # ile epok poczeka zanim przerwie
    restore_best_weights=True  # wróć do najlepszych wag (super ważne!)
    )

    history = model.fit(train_dataset, train_labels, epochs=200, validation_split=0.2, verbose=1, batch_size=32,  callbacks=[early_stop])
    return history


  def model_plots():
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist['rmse'] = np.sqrt(hist['mse'])
    hist['val_rmse'] = np.sqrt(hist['val_mse'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['mae'], name='mae', mode='markers+lines'))
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_mae'], name='val_mae', mode='markers+lines'))
    fig.update_layout(width=1000, height=500, title='MAE vs. VAL_MAE', xaxis_title='Epoki', yaxis_title='Mean Absolute Error', yaxis_type='log')
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['rmse'], name='rmse', mode='markers+lines'))
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_rmse'], name='val_rmse', mode='markers+lines'))
    fig.update_layout(width=1000, height=500, title='RMSE vs. VAL_RMSE', xaxis_title='Epoki', yaxis_title='Root Mean Squared Error', yaxis_type='log')
    fig.show()

    return plot_hist(history)




