import matplotlib.cm as cm

def backtest_signal(monthly_return, predictions):
  df_predictions = predictions.df_predictions.copy(deep=True)
  if not predictions.classification:
    prediction_diff = df_predictions['prediction'].diff()
    df_predictions.insert(len(df_predictions.columns), 'signal',
                          prediction_diff.apply(lambda x : 1 if x <= 0 else -1), allow_duplicates=True)
    df_predictions = df_predictions.dropna()
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

  fig = plt.figure(figsize = (25,10))
  ax = fig.add_subplot(111)
  X = predictions_list[0].df_predictions.index
  colors = cm.rainbow(np.linspace(0, 1, len(predictions_list)))
  labels = []
  for color, prediction in zip(colors, predictions_list):
    df_return = backtest_signal(monthly_return, prediction)
    y = df_return[['cum_return']]
    annual_return = round(df_return["strategy_return"].mean() * 12, 4)
    labels.append(f'{prediction.model_name} (AR = {annual_return * 100} %)')
    y.plot(y=['cum_return'], color=color, ax=ax)
  ax.set_ylabel('Return')
  ax.legend(labels, frameon=False)
  plt.grid()