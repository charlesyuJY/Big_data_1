import os
import pickle
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from tqdm.notebook import tqdm
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, KFold
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def model_filenames(models_folder_path, model_type, model_name, train_size, cv_type, cv_size, iter):
  model_name = model_name.lower().replace(' ', '_')
  prefix = f'model_{model_name}_train{train_size}_cv{cv_type}_{cv_size}'
  model_folder = f'{models_folder_path}/{model_type}/{prefix}'
  model_filename = f'{prefix}_iter{iter}.pkl'
  return model_folder, model_filename

def load_model(models_folder_path, model_type, model_name,
               train_size, cv_type, cv_size, iter):
  model_folder, model_filename = model_filenames(models_folder_path, model_type,
                                                 model_name, train_size, cv_type,
                                                 cv_size, iter)

  model = None
  if os.path.exists(f'{model_folder}/{model_filename}'):
    with open(f'{model_folder}/{model_filename}', 'rb') as f:
      model = pickle.load(f)
  
  return model

def store_model(model, models_folder_path, model_type, model_name,
                train_size, cv_type, cv_size, iter):
  model_folder, model_filename = model_filenames(models_folder_path, model_type,
                                                 model_name, train_size, cv_type,
                                                 cv_size, iter)

  if not os.path.isdir(model_folder):
    os.mkdir(model_folder)
  with open(f'{model_folder}/{model_filename}', 'wb') as f:
    pickle.dump(model, f)

class Prediction:

    def __init__(self, df_predictions, model_path, model_name, model_type,
                 train_size, cv_type, cv_size, param_grid,
                 param_grid_iter):
      self.df_predictions = df_predictions
      self.model_path = model_path
      self.model_name = model_name
      self.model_type = model_type
      self.train_size = train_size
      self.cv_type = cv_type
      self.cv_size = cv_size
      self.param_grid = param_grid
      self.param_grid_iter = param_grid_iter

    def r2_oos(self):
        return r2_score(self.df_predictions["actual"], self.df_predictions["prediction"])
    
    def r2_is(self):
        return self.df_predictions['R2 IS']

    def plot(self, plot_params):
      if self.param_grid is not None:
        numeric_params = [p for p in self.param_grid.keys() if is_numeric_dtype(self.df_predictions[p])]
      else:
        numeric_params = []
      
      columns = 3
      rows = len(numeric_params) // columns + 1

      fig, axs = plt.subplots(rows + int(self.param_grid is not None), columns, figsize=(25, 5 * rows))
      fig.tight_layout()
      fig.subplots_adjust(hspace=0.5)
      self.df_predictions.plot(y=['actual', 'prediction'], title=self.model_name, ax=(axs[0, 0] if len(numeric_params) > 0 else axs[0]))
      self.df_predictions.plot(y=['R2_IS'], title=self.model_name, ax=(axs[0, 1] if len(numeric_params) > 0 else axs[1]))
      self.df_predictions.plot(y=['prediction RMSE', 'benchmark RMSE', 'delta RMSE'], title=self.model_name, ax=(axs[0, 2] if len(numeric_params) > 0 else axs[2]))
      if len(numeric_params) > 0:
        for i, param in enumerate(numeric_params):
          self.df_predictions.plot(y=[param], title=f'{self.model_name} Hyperparameter: {param}', ax=axs[(i + 3) // 3, i % 3])

def rolling_mse(y1, y2):
    mse = (y1 - y2) ** 2
    mse = mse.expanding(1).mean().dropna()
    return mse

def benchmark_expanding(y):
  return y.expanding(1).mean().shift(1).dropna()

def benchmark_rolling(y, window_size):
  return y.rolling(window_size).mean().shift(1).dropna()

def benchmark_compare(y_actual, y_pred, benchmark, window_size=None):
  if benchmark == 'rolling' and window_size is not None:
    y_benchmark = benchmark_rolling(y_actual, window_size)
  elif benchmark == 'expanding':
    y_benchmark = benchmark_expanding(y_actual)
  else:
    raise ValueError(
        """
        Provide a valid value for the benchmark variable ('rolling' or 'expanding').
        For 'rolling', a window size should be provided
        """)
                 
  df_comparison = y_pred.merge(y_benchmark, left_index=True, right_index=True) \
                        .merge(y_actual, left_index=True, right_index=True)
  df_comparison = df_comparison.rename(columns={
      old: new for old, new in zip(df_comparison.columns, ['prediction', 'benchmark', 'actual'])
  })
  df_comparison = df_comparison.dropna()
  
  df_comparison['prediction RMSE'] = np.sqrt(rolling_mse(df_comparison['prediction'], df_comparison['actual']))
  df_comparison['benchmark RMSE'] = np.sqrt(rolling_mse(df_comparison['benchmark'], df_comparison['actual']))
  df_comparison['delta RMSE'] = df_comparison['benchmark RMSE'] - df_comparison['prediction RMSE']

  return df_comparison


def train_model_cv(df_X, df_y, model, models_folder_path, model_type, model_name,
                   train_size, cv_type=None, cv_size=None, param_grid=None,
                   param_grid_iter=20, verbose=True, force_save=False):
  if model_type not in ['rolling', 'expanding']:
      raise ValueError('Type of model should be either "rolling" or "expanding"')
  is_rolling = model_type == 'rolling'
  macro_target = df_y.columns[0]

  date_index = df_X.index
  X = df_X.reset_index(drop=True)
  y = df_y.values.flatten()

  df_predictions = pd.DataFrame({'year_month': [], 'actual': [], 'prediction': [], 'R2_IS': []})
  tscv = TimeSeriesSplit(gap=0, max_train_size=train_size if is_rolling else None, n_splits=X.shape[0] - train_size, test_size=1)
  tscv_split = tqdm(tscv.split(X)) if verbose else tscv.split(X)
  for i, (train_index, validate_index) in enumerate(tscv_split):
    # print("TRAIN:", train_index, "VALIDATE:", validate_index)
    X_train, X_test = X.loc[train_index,].reset_index(drop=True), X.loc[validate_index].reset_index(drop=True)
    y_train, y_test = y[train_index], y[validate_index]

    cv = None
    if cv_type == 'KFold':
      cv = KFold(n_splits=5, shuffle=False)
    elif cv_type == 'TSCV':
      cv = TimeSeriesSplit(gap=0, max_train_size=cv_size if is_rolling else None, n_splits=train_size - cv_size, test_size=1)

    # If no cross-validation is specified, train the model with all training data at once
    if cv is not None:
      model_cv = RandomizedSearchCV(estimator = model, param_distributions = param_grid,
                                    scoring = 'neg_mean_squared_error', cv = cv,
                                    n_iter = param_grid_iter, n_jobs = -1)
    else:
      model_cv = model

    loaded_model = load_model(models_folder_path, model_type, model_name, train_size, cv_type, cv_size, i)
    if loaded_model is None or force_save:
      if verbose:
          tscv_split.set_description(f'Training model')
      model_cv.fit(X_train, y_train)
      store_model(model_cv, models_folder_path, model_type, model_name, train_size, cv_type, cv_size, i)
    else:
      if verbose:
        tscv_split.set_description(f'Using stored model')
      model_cv = loaded_model
    
    r2_is = model_cv.score(X_train, y_train)
    y_pred = model_cv.predict(X_test)

    df_params = {}
    if param_grid is not None:
      for param in param_grid.keys():
        df_params[param] = [model_cv.best_params_[param]]

    df_result = pd.DataFrame({**{'year_month': [date_index[validate_index][0]],
                                 'actual': [y_test[0]],
                                 'prediction': [y_pred[0]],
                                 'R2 IS': [r2_is]},
                              **df_params})
    
    df_predictions = pd.concat([df_predictions, df_result], ignore_index=True)

  df_predictions = df_predictions.set_index('year_month')
  df_benchmark = benchmark_compare(df_predictions['actual'], df_predictions["prediction"],
                                   model_type, window_size=train_size if is_rolling else None)
  df_predictions['prediction RMSE'] = df_benchmark['prediction RMSE']
  df_predictions['benchmark RMSE'] = df_benchmark['benchmark RMSE']
  df_predictions['delta RMSE'] = df_benchmark['delta RMSE']

  return df_predictions
