from .data_source import *
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def backtest_signal(monthly_return, predictions):
  df_predictions = predictions.df_predictions.copy(deep=True)
  if not predictions.classification:
    prediction_diff = df_predictions['prediction'].diff()
    df_predictions = df_predictions.merge(prediction_diff.apply(lambda x : 1 if x <= 0 else -1), left_index=True, right_index=True)
    df_predictions = df_predictions.rename(columns={'prediction_x': 'prediction', 'prediction_y': 'signal'})
  else:
    df_predictions.insert(len(df_predictions.columns), 'signal',
                          df_predictions['prediction'].apply(lambda x : -1 if x == 1 else 1), allow_duplicates=True)

  df_predictions.index = df_predictions.index.astype(str)
  return_df = pd.merge(monthly_return, df_predictions[['signal']], on = "year_month")
  return_df['strategy_return'] = return_df['return'] * return_df['signal']
  return_df['cum_return'] = (1 + return_df.strategy_return).cumprod() - 1

  return return_df

def backtest_plot(macro_target, predictions_list):
  monthly_return = etf_data(macro_target)
  monthly_return = monthly_return[(monthly_return.index >= '2018-08') & (monthly_return.index < '2021-01')]

  fig = plt.figure(figsize = (25,10))
  ax = fig.add_subplot(111)
  X = predictions_list[0].df_predictions.index
  colors = cm.rainbow(np.linspace(0, 1, len(predictions_list) + 1))

  target_cum_return = ((1 + monthly_return).cumprod() - 1) * 100
  annual_return = monthly_return['return'].mean() * 12
  target_cum_return.plot(color=colors[0], ax=ax)
  if macro_target.lower() == 'aaa':
    target_label = 'iShares Aaa - A Rated Corporate Bond ETF'
  else:
    target_label = 'iShares U.S. Treasury Bond ETF'
  labels = [f'{target_label} (AR = {round(annual_return * 100, 2)} %)']
  for color, prediction in zip(colors[1:], predictions_list):
    df_return = backtest_signal(monthly_return, prediction)
    y = df_return[['cum_return']] * 100
    annual_return = df_return["strategy_return"].mean() * 12
    labels.append(f'{prediction.model_name} {prediction.model_type} {prediction.cv_type} (AR = {round(annual_return * 100, 2)} %)')
    y.plot(y=['cum_return'], color=color, ax=ax)
  ax.set_ylabel('Return (%)')
  ax.set_xlabel('Month')
  ax.legend(labels, frameon=False)
  plt.grid()
