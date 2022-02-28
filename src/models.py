import os
import pickle
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from tqdm.notebook import tqdm
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, KFold
from sklearn.metrics import r2_score, accuracy_score, recall_score, precision_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess

def model_filenames(models_folder_path, model_type, model_name, train_size, cv_type, cv_size, iter, classification=False):
  model_name = model_name.lower().replace(' ', '_')
  prefix = f'model_{model_name}_train{train_size}_cv{cv_type}_{cv_size}'
  prediction_type = 'classification' if classification else 'regression'
  model_folder = f'{models_folder_path}/{model_type}/{prediction_type}/{prefix}'
  model_filename = f'{prefix}_iter{iter}.pkl'
  return model_folder, model_filename

def load_model(models_folder_path, model_type, model_name,
               train_size, cv_type, cv_size, iter, classification=False):
  model_folder, model_filename = model_filenames(models_folder_path, model_type,
                                                 model_name, train_size, cv_type,
                                                 cv_size, iter, classification)

  model = None
  if os.path.exists(f'{model_folder}/{model_filename}'):
    with open(f'{model_folder}/{model_filename}', 'rb') as f:
      model = pickle.load(f)
  
  return model

def store_model(model, models_folder_path, model_type, model_name,
                train_size, cv_type, cv_size, iter, classification=False):
  model_folder, model_filename = model_filenames(models_folder_path, model_type,
                                                 model_name, train_size, cv_type,
                                                 cv_size, iter, classification)

  prediction_type = 'classification' if classification else 'regression'
  if not os.path.isdir(f'{models_folder_path}/{model_type}'):
    os.mkdir(f'{models_folder_path}/{model_type}')
  if not os.path.isdir(f'{models_folder_path}/{model_type}/{prediction_type}'):
    os.mkdir(f'{models_folder_path}/{model_type}/{prediction_type}')
  if not os.path.isdir(model_folder):
    os.mkdir(model_folder)
  with open(f'{model_folder}/{model_filename}', 'wb') as f:
    pickle.dump(model, f)

class PredictionSet:
  def __init__(self, prediction_set, regressor_names):
    self.prediction_set = prediction_set
    self.regressor_names = regressor_names
  
  def plot(self):
    fig, axs = plt.subplots(len(self.prediction_set), 3, figsize=(25, 70))
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5)

    for i, prediction in enumerate(self.prediction_set):
      prediction.df_predictions.plot(y=['actual', 'prediction'], title=prediction.model_name, ax=axs[i, 0])
      prediction.df_predictions.plot(y=['R2 IS'], title=prediction.model_name, ax=axs[i, 1])
      prediction.df_predictions.plot(y=['prediction RMSE', 'benchmark RMSE', 'delta RMSE'], title=prediction.model_name, ax=axs[i, 2])

  def model_report(self):
    df_report = pd.DataFrame({'Predictor Name': [], 'R2 IS': [], 'R2 OOS': [], 'delta RMSE':[]})

    for i, predictor in enumerate(self.prediction_set):
      df_report = pd.concat([df_report, pd.DataFrame({'Predictor Name': [self.regressor_names[i]],
                                                      'R2 IS': [predictor.r2_is().mean()],
                                                      'R2 OOS': [predictor.r2_oos()],
                                                      'delta RMSE': [predictor.delta_rmse()]})])
    return df_report.set_index('Predictor Name').sort_values(by='R2 OOS', ascending=False)

class Prediction:

    def __init__(self, df_predictions, models_folder_path, model_name, model_type,
                 train_size, cv_type, cv_size, param_grid, param_grid_iter, classification=False):
      self.df_predictions = df_predictions
      self.model_name = model_name
      self.classification = classification
      self.model_type = model_type
      self.train_size = train_size
      self.cv_type = cv_type
      self.cv_size = cv_size
      self.param_grid = param_grid
      self.param_grid_iter = param_grid_iter

      self.model_path = model_filenames(models_folder_path, model_type, model_name,
                                        train_size, cv_type, cv_size, 0, classification)[0]
      
      self.save_performance(models_folder_path)

    def r2_oos(self):
      if self.classification:
        return None
      return r2_score(self.df_predictions["actual"], self.df_predictions["prediction"])
    
    def r2_is(self):
      if self.classification:
        return None
      return self.df_predictions['R2 IS']

    def delta_rmse(self):
      if self.classification:
        return None
      return self.df_predictions['delta RMSE'].values[-1]
    
    def accuracy(self):
      if not self.classification:
        return None
      return self.df_predictions['prediction accuracy'].values[-1]

    def auc(self):
      if not self.classification:
        return None
      return self.df_predictions['prediction AUC'].values[-1]
    
    def precision(self):
      if not self.classification:
        return None
      return self.df_predictions['prediction precision'].values[-1]
    
    def recall(self):
      if not self.classification:
        return None
      return self.df_predictions['prediction recall'].values[-1]


    def model_report(self):
      if self.classification:
        return pd.DataFrame({
            'Accuracy': [self.accuracy()],
            'AUC': [self.auc()],
            'precision': [self.precision()],
            'recall': [self.recall()]
        })
      else:
        return pd.DataFrame({
            'R2 OOS': [self.r2_oos()],
            'R2 IS': [self.r2_is().mean()]
            })

    def save_performance(self, models_folder_path):
      with open(f'{models_folder_path}/performance.pkl', 'rb') as f:
        performance_dict = pickle.load(f)
      performance_dict[self.model_path] = {
          'model_name': self.model_name,
          'classification': self.classification,
          'model_type': self.model_type,
          'train_size': self.train_size,
          'cv_type': self.cv_type,
          'cv_size': self.cv_size,
          'param_grid': self.param_grid,
          'param_grid_iter': self.param_grid_iter,
          'r2_oos': self.r2_oos(),
          'r2_is': None if self.classification else self.r2_is().mean(),
          'delta_rmse': self.delta_rmse(),
          'accuracy': self.accuracy(),
          'auc': self.auc(),
          'precision': self.precision(),
          'recall': self.recall(),
          'prediction': self
      }
      with open(f'{models_folder_path}/performance.pkl', 'wb') as f:
        loaded_dict = pickle.dump(performance_dict, f)
      
      subprocess.run(['git', 'add', f'{models_folder_path}/*'])
      subprocess.run(['git', 'commit', '-m', 'AUTOMATED UPLOAD: Update models'])
      subprocess.run(['git', 'push'])

    def plot(self):
      if self.param_grid is not None:
        numeric_params = [p for p in self.param_grid.keys() if is_numeric_dtype(self.df_predictions[p])]
      else:
        numeric_params = []
      
      columns = 3
      rows = len(numeric_params) // columns + 1

      fig, axs = plt.subplots(rows + int(self.param_grid is not None), columns, figsize=(25, 5 * rows))
      fig.tight_layout()
      fig.subplots_adjust(hspace=0.5)      

      if self.classification:
        cf_matrix = confusion_matrix(self.df_predictions['actual'], self.df_predictions['prediction'])
        cf_ax = axs[0, 0] if len(numeric_params) > 0 else axs[0]
        sns.heatmap(cf_matrix, annot=True, cmap='Blues', ax=cf_ax)
        cf_ax.set_title(self.model_name);
        cf_ax.set_xlabel('Predicted Values')
        cf_ax.set_ylabel('Actual Values')
        cf_ax.xaxis.set_ticklabels(['Down','Up'])
        cf_ax.yaxis.set_ticklabels(['Down','Up'])

        self.df_predictions.plot(y=['AUC IS', 'precision IS', 'recall IS'], title=self.model_name, ax=(axs[0, 1] if len(numeric_params) > 0 else axs[1]))

        fpr, tpr, _ = roc_curve(self.df_predictions['actual'], self.df_predictions['prediction'])
        roc_ax = axs[0, 2] if len(numeric_params) > 0 else axs[2]
        roc_ax.plot(fpr, tpr, color="darkorange", lw=2,label="ROC curve (area = %0.2f)" % self.auc())
        roc_ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        roc_ax.set_xlabel("False Positive Rate")
        roc_ax.set_ylabel("True Positive Rate")
        roc_ax.set_title(self.model_name)
        roc_ax.legend(loc="lower right")
      else:
        self.df_predictions.plot(y=['actual', 'prediction'], title=self.model_name, ax=(axs[0, 0] if len(numeric_params) > 0 else axs[0]))
        self.df_predictions.plot(y=['R2 IS'], title=self.model_name, ax=(axs[0, 1] if len(numeric_params) > 0 else axs[1]))
        self.df_predictions.plot(y=['prediction RMSE', 'benchmark RMSE', 'delta RMSE'], title=self.model_name, ax=(axs[0, 2] if len(numeric_params) > 0 else axs[2]))
      if len(numeric_params) > 0:
        for i, param in enumerate(numeric_params):
          self.df_predictions.plot(y=[param], title=f'{self.model_name} Hyperparameter: {param}', ax=axs[(i + 3) // 3, i % 3])

    @staticmethod
    def find_models(models_folder_path, model_name=None, classification=None, model_type=None,
                         train_size=None, cv_type=None, cv_size=None, param_grid_iter=None,
                         r2_oos=None, r2_is=None, delta_rmse=None, accuracy=None, auc=None, recall=None, precision=None):
      with open(f'{models_folder_path}/performance.pkl', 'rb') as f:
        performance_dict = pickle.load(f)
      
      if model_name is not None:
        performance_dict = {mpn: m for mpn, m in performance_dict.items()
                            if m['model_name'] == model_name}
      if classification is not None:
        performance_dict = {mpn: m for mpn, m in performance_dict.items()
                            if m['classification'] == classification}
      if model_type is not None:
        performance_dict = {mpn: m for mpn, m in performance_dict.items()
                            if m['model_type'] == model_type}
      if train_size is not None:
        performance_dict = {mpn: m for mpn, m in performance_dict.items()
                            if m['train_size'] == train_size}
      if cv_type is not None:
        performance_dict = {mpn: m for mpn, m in performance_dict.items()
                            if m['cv_type'] == cv_type}
      if cv_size is not None:
        performance_dict = {mpn: m for mpn, m in performance_dict.items()
                            if m['cv_size'] == cv_size}
      if param_grid_iter is not None:
        performance_dict = {mpn: m for mpn, m in performance_dict.items()
                            if m['param_grid_iter'] == param_grid_iter}
      if r2_oos is not None:
        performance_dict = {mpn: m for mpn, m in performance_dict.items()
                            if m['r2_oos'] >= r2_oos}
      if r2_is is not None:
        performance_dict = {mpn: m for mpn, m in performance_dict.items()
                            if m['r2_is'] >= r2_is}
      if delta_rmse is not None:
        performance_dict = {mpn: m for mpn, m in performance_dict.items()
                            if m['delta_rmse'] >= delta_rmse}
      if accuracy is not None:
        performance_dict = {mpn: m for mpn, m in performance_dict.items()
                            if m['accuracy'] >= accuracy}
      if auc is not None:
        performance_dict = {mpn: m for mpn, m in performance_dict.items()
                            if m['auc'] >= auc}
      if precision is not None:
        performance_dict = {mpn: m for mpn, m in performance_dict.items()
                            if m['precision'] >= precision}
      if recall is not None:
        performance_dict = {mpn: m for mpn, m in performance_dict.items()
                            if m['recall'] >= recall}
      
      return performance_dict

def rolling_mse(y1, y2):
    mse = (y1 - y2) ** 2
    mse = mse.expanding(1).mean().dropna()
    return mse

def benchmark_expanding(y, classification=False):
  y_result = y.expanding(1)

  if classification:
    y_result = y_result.apply(lambda x: x.mode()[0])
  else:
    y_result = y_result.mean()

  return y_result.shift(1).dropna()

def benchmark_rolling(y, window_size, classification=False):
  y_result = y.rolling(window_size)
  
  if classification:
    y_result = y_result.apply(lambda x: x.mode()[0])
  else:
    y_result = y_result.mean()

  return y_result.shift(1).dropna()

def benchmark_compare(y_actual, y_pred, benchmark, classification=False, window_size=None):
  if benchmark == 'rolling' and window_size is not None:
    y_benchmark = benchmark_rolling(y_actual, window_size, classification)
  elif benchmark == 'expanding':
    y_benchmark = benchmark_expanding(y_actual, classification)
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
  
  if classification:
    df_comparison['prediction accuracy'] = accuracy_score(df_comparison['actual'], df_comparison['prediction'])
    df_comparison['prediction AUC'] = roc_auc_score(df_comparison['actual'], df_comparison['prediction'])
    df_comparison['prediction precision'] = precision_score(df_comparison['actual'], df_comparison['prediction'])
    df_comparison['prediction recall'] = recall_score(df_comparison['actual'], df_comparison['prediction'])
    df_comparison['benchmark accuracy'] = accuracy_score(df_comparison['actual'], df_comparison['benchmark'])
    df_comparison['benchmark AUC'] = roc_auc_score(df_comparison['actual'], df_comparison['benchmark'])
    df_comparison['benchmark precision'] = precision_score(df_comparison['actual'], df_comparison['benchmark'])
    df_comparison['benchmark recall'] = recall_score(df_comparison['actual'], df_comparison['benchmark'])
    df_comparison['delta AUC'] = df_comparison['benchmark AUC'] - df_comparison['prediction AUC']
  else:
    df_comparison['prediction RMSE'] = np.sqrt(rolling_mse(df_comparison['prediction'], df_comparison['actual']))
    df_comparison['benchmark RMSE'] = np.sqrt(rolling_mse(df_comparison['benchmark'], df_comparison['actual']))
    df_comparison['delta RMSE'] = df_comparison['benchmark RMSE'] - df_comparison['prediction RMSE']

  return df_comparison

def train_model_cv(df_X, df_y, model, models_folder_path, model_type, model_name,
                   train_size, cv_type=None, cv_size=None, param_grid=None,
                   param_grid_iter=20, classification=False, verbose=True, force_save=False):
  if model_type not in ['rolling', 'expanding']:
      raise ValueError('Type of model should be either "rolling" or "expanding"')
  is_rolling = model_type == 'rolling'
  macro_target = df_y.columns[0]

  date_index = df_X.index
  X = df_X.reset_index(drop=True)
  y = df_y.values.flatten()

  df_predictions = pd.DataFrame({})
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
      scoring = 'roc_auc' if classification else 'neg_mean_squared_error'
      model_cv = RandomizedSearchCV(estimator = model, param_distributions = param_grid,
                                    scoring = scoring, cv = cv,
                                    n_iter = param_grid_iter, n_jobs = -1)
    else:
      model_cv = model

    loaded_model = load_model(models_folder_path, model_type, model_name, train_size, cv_type, cv_size, i, classification)
    if loaded_model is None or force_save:
      if verbose:
          tscv_split.set_description(f'Training model')
      model_cv.fit(X_train, y_train)
      store_model(model_cv, models_folder_path, model_type, model_name, train_size, cv_type, cv_size, i, classification)
    else:
      if verbose:
        tscv_split.set_description(f'Using stored model')
      model_cv = loaded_model
    
    y_pred = model_cv.predict(X_test)
    y_pred_is = model_cv.predict(X_train)

    if classification:
      y_pred = y_pred >= 0.5
      y_pred_is = y_pred_is >= 0.5
      accuracy_is = {
          'accuracy IS': [accuracy_score(y_train, y_pred_is)],
          'AUC IS': [roc_auc_score(y_train, y_pred_is)],
          'precision IS': [precision_score(y_train, y_pred_is)],
          'recall IS': [recall_score(y_train, y_pred_is)],
      }
    else:
      accuracy_is = {
          'R2 IS': [model_cv.score(X_train, y_train)]
      }

    df_params = {}
    if param_grid is not None:
      for param in param_grid.keys():
        df_params[param] = [model_cv.best_params_[param]]

    df_result = pd.DataFrame({**{'year_month': [date_index[validate_index][0]],
                                  'actual': [y_test[0]],
                                  'prediction': [y_pred[0]]},
                              **accuracy_is,
                              **df_params})
    
    df_predictions = pd.concat([df_predictions, df_result], ignore_index=True)

  df_predictions = df_predictions.set_index('year_month')
  df_benchmark = benchmark_compare(pd.DataFrame({macro_target: y}).set_index(date_index), df_predictions[['prediction']],
                                   model_type, window_size=train_size if is_rolling else None,
                                   classification=classification)
  
  if classification:
    benchmark_cols = ['prediction accuracy', 'prediction AUC', 'prediction precision', 'prediction recall',
                      'benchmark accuracy', 'benchmark AUC', 'benchmark precision', 'benchmark recall', 'delta AUC']
  else:
    benchmark_cols = ['prediction RMSE', 'benchmark RMSE', 'delta RMSE']
  df_predictions = df_predictions.merge(df_benchmark[benchmark_cols],
                                        left_index=True, right_index=True)

  return Prediction(df_predictions, models_folder_path, model_name, model_type, train_size,
                    cv_type, cv_size, param_grid, param_grid_iter, classification=classification)

def train_multiple_cv(df_X, df_y, model, models_folder_path, model_type, model_name,
                      train_size, regressors, cv_type=None, cv_size=None, param_grid=None,
                      param_grid_iter=20, verbose=True, force_save=False, regressor_names=None, classification=False):
  
  predictions = []
  iterator = tqdm(regressors) if verbose else regressors
  for regressor in iterator:
    if verbose:
      # print(model_filenames(models_folder_path, model_type, model_name, train_size, cv_type, cv_size, 0)[0])
      if os.path.exists(model_filenames(models_folder_path, model_type, model_name, train_size, cv_type, cv_size, 0, classification)[0]):
        iterator.set_description(f'Loading model from file for {regressor}')
      else:
        iterator.set_description(f'Training model for {regressor}')
    
    predictions.append(train_model_cv(df_X[[regressor]], df_y, model, models_folder_path, model_type,
                                      f'{model_name} {regressor}', train_size, cv_type, cv_size, param_grid,
                                      param_grid_iter, False, force_save, classification=classification))

  return PredictionSet(predictions, regressors if regressor_names is None else regressor_names)