def data_clean():
  import pandas as pd
  dataset = raw_dataset.copy()
  dataset.dropna(inplace=True)
  index_to_drop = dataset[dataset['median_house_value'] == 500001]
  dataset = dataset.drop(index=index_to_drop.index)
  dataset_dummies = pd.get_dummies(dataset, drop_first=True)
  return dataset_dummies

def data_division():
  train_dataset = dataset_dummies.sample(frac=0.8, random_state=0)
  test_dataset = dataset_dummies.drop(train_dataset.index)
  train_labels = train_dataset.pop('median_house_value')
  test_labels = test_dataset.pop('median_house_value')

  from sklearn.preprocessing import StandardScaler

  scaler = StandardScaler()
  train_dataset_scaled = pd.DataFrame(data = scaler.fit_transform(train_dataset), columns=train_dataset.columns)
  test_dataset_scaled = pd.DataFrame(data = scaler.transform(test_dataset), columns=test_dataset.columns)

  return train_dataset_scaled, train_labels, test_dataset_scaled, test_labels
  

  
