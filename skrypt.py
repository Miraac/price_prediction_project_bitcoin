import pandas as pd
import numpy as np
#pobieranie danych
frame = pd.read_csv('C:\projekty\kruptobot\BTCUSD.csv')

fgiee = pd.read_csv('nastroje.csv')


#tworzenie macd
#kom - pozycja indeksu
kom = 30
#test
#mad = (sum(frame.loc[0-kom:kom,'Close'])/10) - (sum(frame.loc[0-kom:kom,'Close'])/30)

#pętla tworząca wskaźnik macd 10,30
for i in range(kom,len(frame)):
    mad = (sum(frame.loc[i-10:i,'Close'])/10) - (sum(frame.loc[i-30:i,'Close'])/30)
    frame.loc[i,'macd']=mad
    

#pętla tworząca wskaźnik rsi 14 close    
def rsi(prices, n=14):
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)

    for i in range(n, len(prices)):
        delta = deltas[i-1] # cause the diff is 1 shorter

        if delta>0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)

    return rsi

frame['rsi']= rsi(frame['Close'])

for i in range(kom,len(frame)):
    if frame.loc[i,'rsi'] > 70:
        frame.loc[i,'rsisig'] = 1
    elif frame.loc[i,'rsi'] < 30:
        frame.loc[i,'rsisig'] = -1
    else: 
        frame.loc[i,'rsisig'] = 0
y=pd.DataFrame()
# kolumny dot % wartoci otwarcia/zamknięcia oraz zasięgu dniowego

frame['diff_perc'] = frame['Close']/frame['Open']*100-100
frame['range_perc']= frame['High']/frame['Low']*100-100

# usuwanie wartoci niepełych z początku - zawierających NaN
frame.dropna(how='any',inplace=True)

#ustawianie indeksu jako data
frame.set_index('Date',inplace=True)
frame.reset_index(inplace=True)

#ustawianie y jako cena zamknięcia dla kolejnego dnia
i=len(frame)
for l in range (0,i-1):
    y.loc[l,'pred'] = frame.loc[l+1,'Close']
        
#dodawanie fear & greed
fgiee['newdate'] = fgiee['date'].astype(str).str[6:10] + fgiee['date'].astype(str).str[2:5] +'-'+ fgiee['date'].astype(str).str[0:2]
fgiee.drop('date',axis=1,inplace=True)
full = frame.merge(fgiee, left_on='Date', right_on='newdate')

print(frame.tail())
#def nastroj(x)

import tensorflow as tf
import numpy as np



# Podziel dane na cechy (X) i etykiety (y)
X = frame
X.drop(columns='Date',inplace=True)
y.tail()
X.tail()

#dodawanie brakującej wartości y dla dnia następnego
y.loc[2984,] = 16446

##TENSORFLOW WLASCIWY
# Utwórz warstwę wejściową o rozmiarze 8 (odpowiadającą 8 cechom)
input_layer = tf.keras.layers.Input(shape=(11,))

# Utwórz ukrytą warstwę z 64 neuronami z aktywacją ReLU
hidden_layer = tf.keras.layers.Dense(64, activation="relu")(input_layer)

# Utwórz warstwę wyjściową z jednym neuronem z aktywacją liniową
output_layer = tf.keras.layers.Dense(1, activation="linear")(hidden_layer)

# Utwórz model z warstwami wejściową, ukrytą i wyjściową
model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

# Kompiluj model z użyciem optymalizatora Adam i funkcji straty MSE
model.compile(optimizer="adam", loss="mean_squared_error")

# Trenuj model na danych treningowych
model.fit(X, y, epochs=10)


###2.0



# Generuj przewidywania dla nowych danych
predictions = model.predict(X)

# Porównaj przewidywania z rzeczywistymi etykietami
for prediction, actual in zip(predictions, y):
  print("Prediction:", prediction, "Actual:", actual)
