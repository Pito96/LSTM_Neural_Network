# import knižníc
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential

# načítanie dát
podnik = 'TSLA'

zaciatocne_data = dt.datetime(2012, 1, 1)
koncove_data = dt.datetime(2021, 6, 1)

df = web.DataReader(podnik, 'yahoo', zaciatocne_data, koncove_data)

# print(df.head(), '\n')
# print('V danej tabuľke(dataframe) sa nachádza: ', df.shape[1], 'stĺpcov a v každom z nich je ', df.shape[0], 'hodnôt')
# print('Počet Null(prázdnych) hodnôt v tabuľke: ', df.isnull().sum().sum())
#
# fig, ax = plt.subplots(figsize=(16, 8))
# ax.set_facecolor('white')
# ax.plot(df['Close'], color='blue', label=f"Skutočná cena {podnik}")
# plt.title(f"Reálna cena {podnik} ")
# plt.xlabel('Čas')
# plt.ylabel(f"Cena akcií {podnik} ")
# plt.legend()
# plt.show()

# príprava dát
data = df.filter(['Close'])
dataset = data.values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
skalovane_data = scaler.fit_transform(dataset)

# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

dni_pre_predikciu = 60

x_trenovacie_data = []
y_trenovacie_data = []

for x in range(dni_pre_predikciu, len(skalovane_data)):
    x_trenovacie_data.append(skalovane_data[x - dni_pre_predikciu:x, 0])
    y_trenovacie_data.append(skalovane_data[x, 0])

x_trenovacie_data = np.array(x_trenovacie_data)  # z viacerých 1D polí vytvoríme jedno 2D pole pomocou NumPy
y_trenovacie_data = np.array(y_trenovacie_data)

x_trenovacie_data = np.reshape(x_trenovacie_data, (x_trenovacie_data.shape[0], x_trenovacie_data.shape[1], 1))

# vytvorenie modelu
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_trenovacie_data.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # predikcia ďalšej uzatváracej ceny

# trénovanie modelu
model.compile(optimizer='adam', loss='mean_squared_error', metrics='mean_absolute_percentage_error')
history = model.fit(x_trenovacie_data, y_trenovacie_data, epochs=30, batch_size=29, validation_split=0.1)

# zobrazenie grafu vykonnosti
plt.plot(history.history['val_mean_absolute_percentage_error'])
plt.title('Výkonnosť modelu')
plt.ylabel('Percentuály rozdiel predikovanej a skutočnej ceny')
plt.xlabel('Epocha')
plt.legend(['Validácia'], loc='upper right')
plt.show()

# testovanie modelu na existujúcich dátach

# načítanie testovacích dát
zaciatocne_test_data = dt.datetime(2021, 6, 1)
koncove_test_data = dt.datetime(2022, 2, 17)

test_data = web.DataReader(podnik, 'yahoo', zaciatocne_test_data, koncove_test_data)
sucasne_data = test_data['Close'].values
celkove_data = pd.concat((df['Close'], test_data['Close']), axis=0)  # zlúčenie riadkov 2 dataframeov pomocou axis=0

vstupy_modelu = celkove_data[len(celkove_data) - len(test_data) - dni_pre_predikciu:].values
vstupy_modelu = vstupy_modelu.reshape(-1, 1)
vstupy_modelu = scaler.transform(vstupy_modelu)

# vytvorenie predikcie na testovacich datach
x_testovacie_data = []

for x in range(dni_pre_predikciu, len(vstupy_modelu) + 1):
    x_testovacie_data.append(vstupy_modelu[x - dni_pre_predikciu:x, 0])

x_testovacie_data = np.array(x_testovacie_data)
x_testovacie_data = np.reshape(x_testovacie_data, (x_testovacie_data.shape[0], x_testovacie_data.shape[1], 1))

predikovane_ceny = model.predict(x_testovacie_data)
predikovane_ceny = scaler.inverse_transform(predikovane_ceny)

# vykreslenie predikcii
fig, ax = plt.subplots(figsize=(16, 8))
ax.set_facecolor('white')
ax.plot(sucasne_data, color='cyan', label=f"Skutočná cena {podnik}")
ax.plot(predikovane_ceny, color='red', label=f"Predikovaná cena {podnik}")
plt.title(f"Reálna cena {podnik}")
plt.xlabel('Čas v dňoch')
plt.ylabel(f"Cena akcií {podnik} v $ ")
plt.legend()
plt.show()

#PREDIKCIA NASLEDUJUCEHO DNA
#vytvorenie dataframe z close cien využitím testovacích dát z fázy testovania
novy_df_close = test_data.filter(['Close'])

# zoberiem posledných 60 dní close cien
poslednych_60_dni = novy_df_close[-60:].values
# print(poslednych_60_dni)

# zoscaleujem posledných 60 dní medzi 0 and 1
poslednych_60_dni_scaled = scaler.transform(poslednych_60_dni)

# vytvorím prázdny list pre dáta na ktorých chcem vytvoriť predikciu nasledujúceho dňa
x_test_nasled = []

#priradenie posledych zoscaleovanych 60 dni do x_test_nasled
x_test_nasled.append(poslednych_60_dni_scaled)

#prevedenie x_test_nasled dát do poľa pomocou numpy
x_test_nasled = np.array(x_test_nasled)

#Reshape dát na 3D pole
x_test_nasled = np.reshape(x_test_nasled, (x_test_nasled.shape[0], x_test_nasled.shape[1], 1))

#vytvorenie predikcie nasledujúceho dňa na základe x_test_nasled dát
pred_cena_nasled_dna = model.predict(x_test_nasled)

# spätné scaleovanie hodnoty z intervalu (0,1) na reálnu cenu
pred_cena_nasled_dna = scaler.inverse_transform(pred_cena_nasled_dna)

# výpis ceny na zajtra
print(f"Predikcia ceny akcie {podnik} na zajtra je: {pred_cena_nasled_dna}")

# # predikcia nasledujuceho dna
# real_data = [vstupy_modelu[len(vstupy_modelu) + 1 - dni_pre_predikciu: len(vstupy_modelu + 1), 0]]
# # real_data = scaler.inverse_transform(real_data)
# # print(real_data)
# real_data = np.array(real_data)
# real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
#
# predikcia = model.predict(real_data)
# predikcia = scaler.inverse_transform(predikcia)
# print(f"Predikcia ceny akcie {podnik} na zajtra je: {predikcia}")