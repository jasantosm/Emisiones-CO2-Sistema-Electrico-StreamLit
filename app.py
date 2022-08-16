import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests



def sliding_time(ts, window_size=1):

  n = ts.shape[0] - window_size  
  X = np.empty((n, window_size))
  y = np.empty(n)

  for i in range(window_size, ts.shape[0]):   
    y[i - window_size] = ts[i]
    X[i- window_size, 0:window_size] = np.array(ts[i - window_size:i])
    
  return X, y

def multi_var_slinding_time(df, window_size, objetive_var):
  X_train, y_train = sliding_time(df[objetive_var].values, window_size=window_size)

  for feature in df.columns:
    if feature!=objetive_var:
      X_multi, y_multi = sliding_time(df[feature].values, window_size=window_size)
      X_multi = np.append(X_multi, y_multi.reshape(-1,1), axis=1)
      X_train = np.append(X_train, X_multi, axis=1)

  return X_train, y_train

def main():

    st.set_page_config(page_title='CO2 Emissions Colombian Power System APP', page_icon="⚡️")
    st.title('CO2 Emissions Colombian Power System Machine Learning APP')

    st.write('The following is a test dataset for the model:')

    r = requests.get('http://127.0.0.1:8000/model/data/test/')

    response_json = r.json()

    df_test  = pd.DataFrame(response_json['test_data']).set_index('Date')
    y_test = response_json['y_test']
    k = response_json['k']

    
    st.write(df_test)
    
    x = df_test.index[k:]

    if not st.button('Predict'):
        fig = plt.figure(dpi = 120, figsize = (12, 5))
        plt.plot(x, y_test, ls = "--", label="Valor verdadero (pruebas)")
        plt.title("Predicción vs valores verdaderos (pruebas) - MLPRegressor")
        plt.xticks(np.arange(0, 1000, 100))
        plt.xlabel('Date')
        plt.ylabel('CO2eq Ton')
        st.pyplot(fig)
    else:
        #model = load('./model/perceptron.joblib') 
        #y_pred = model.predict(X_test)
        #fig = plt.figure(dpi = 120, figsize = (12, 5))
        plt.plot(x, y_test, ls = "--", label="Valor verdadero (pruebas)")
        plt.plot(x, y_pred, ls = '-', label="Valor predicho (pruebas)")
        plt.title("Predicción vs valores verdaderos (pruebas) - MLPRegressor")
        plt.xticks(np.arange(0, 1000, 100))
        plt.xlabel('Date')
        plt.ylabel('CO2eq Ton')
        st.pyplot(fig)

    if st.button('Erase'):
        fig.clf()


if __name__=='__main__':
    main()