#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector


# In[2]:


df = pd.read_csv('GOOG (2).csv')
df


# In[3]:


df.shape


# In[4]:


import plotly.graph_objects as go
import matplotlib.pyplot as plt
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close price'))
fig.update_layout(showlegend=True, title='Alphabet Inc. Stock Price 2004-2023')
fig.show()


# train = df.loc[df['Date'] <= '2020-12-24']
# test = df.loc[df['Date'] > '2020-12-24']
# 
# scaler = StandardScaler()
# scaler = scaler.fit(np.array(train['Close']).reshape(-1, 1))

# In[5]:


train = df.loc[
    (df['Date'] >= '2004-12-24') & (df['Date'] <= '2007-06-24') |
    (df['Date'] >= '2007-10-24') & (df['Date'] <= '2008-07-24') |
    (df['Date'] >= '2008-08-24') & (df['Date'] <= '2012-06-24') |
    (df['Date'] >= '2012-06-30') & (df['Date'] <= '2014-05-24') |
    (df['Date'] <= '2014-05-30') & (df['Date'] <= '2017-05-24') |
    (df['Date'] >= '2017-06-01') & (df['Date'] <= '2019-09-30') |
    (df['Date'] >= '2021-07-01') & (df['Date'] <= '2022-04-30')
]

test = df.loc[
    (df['Date'] >= '2020-01-01') & (df['Date'] <= '2021-04-30') |
    (df['Date'] >= '2019-10-24') & (df['Date'] <= '2019-12-24') |
    (df['Date'] >= '2021-05-24') & (df['Date'] <= '2021-06-24') |
    (df['Date'] > '2022-05-01')
]

train.shape, test.shape


# In[6]:


from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Define the scaler
scaler = MinMaxScaler()

# Fit the scaler on the training data and transform the 'Close' column
train['Close'] = scaler.fit_transform(np.array(train['Close']).reshape(-1, 1))

# Transform the 'Close' column in the test data using the fitted scaler
test['Close'] = scaler.transform(np.array(test['Close']).reshape(-1, 1))


# In[7]:


from tensorflow.keras import regularizers

# LSTM Autoencoder
TIME_STEPS = 30

def create_sequences(X, y, time_steps=TIME_STEPS):
    X_out, y_out = [], []
    for i in range(len(X)-time_steps):
        X_out.append(X.iloc[i:(i+time_steps)].values)
        y_out.append(y.iloc[i+time_steps])
    
    return np.array(X_out), np.array(y_out)

X_train, y_train = create_sequences(train[['Close']], train['Close'])
X_test, y_test = create_sequences(test[['Close']], test['Close'])
np.random.seed(21)
tf.random.set_seed(21)

model_lstm = Sequential()
model_lstm.add(LSTM(1024, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2]), kernel_regularizer=regularizers.l2(0.01), return_sequences=True))
model_lstm.add(LSTM(512, activation='tanh', kernel_regularizer=regularizers.l2(0.01), return_sequences=True))
model_lstm.add(LSTM(256, activation='tanh', kernel_regularizer=regularizers.l2(0.01), return_sequences=True))
model_lstm.add(LSTM(128, activation='tanh', kernel_regularizer=regularizers.l2(0.01), return_sequences=True))
model_lstm.add(LSTM(64, activation='tanh', kernel_regularizer=regularizers.l2(0.01), return_sequences=False))
model_lstm.add(RepeatVector(X_train.shape[1]))
model_lstm.add(LSTM(64, activation='tanh', kernel_regularizer=regularizers.l2(0.01), return_sequences=True))
model_lstm.add(LSTM(128, activation='tanh', kernel_regularizer=regularizers.l2(0.01), return_sequences=True))
model_lstm.add(LSTM(256, activation='tanh', kernel_regularizer=regularizers.l2(0.01), return_sequences=True))
model_lstm.add(LSTM(512, activation='tanh', kernel_regularizer=regularizers.l2(0.01), return_sequences=True))
model_lstm.add(LSTM(1024, activation='tanh', kernel_regularizer=regularizers.l2(0.01), return_sequences=True))
model_lstm.add(Dense(X_train.shape[2]))

model_lstm.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00005), loss='mse')

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history_lstm = model_lstm.fit(X_train, y_train, epochs=80, batch_size=64, validation_split=0.1, shuffle=False, callbacks=[early_stopping])


# TIME_STEPS = 30

# In[8]:


import matplotlib.pyplot as plt

# Access the loss values from history_lstm
training_loss = history_lstm.history['loss']
validation_loss = history_lstm.history['val_loss']

# Plot the training and validation loss
epochs = range(1, len(training_loss) + 1)
plt.plot(epochs, training_loss, 'b-', label='Training Loss')
plt.plot(epochs, validation_loss, 'r-', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss for LSTM')
plt.legend()
plt.show()


# In[9]:


model_lstm.summary()


# In[128]:


import matplotlib.pyplot as plt

# Increase model complexity and apply regularization
TIME_STEPS = 30

def create_sequences(X, y, time_steps=TIME_STEPS):
    X_out, y_out = [], []
    for i in range(len(X)-time_steps):
        X_out.append(X.iloc[i:(i+time_steps)].values)
        y_out.append(y.iloc[i+time_steps])
    
    return np.array(X_out), np.array(y_out)

X_train, y_train = create_sequences(train[['Close']], train['Close'])
X_test, y_test = create_sequences(test[['Close']], test['Close'])
np.random.seed(1)
tf.random.set_seed(1)

encoding_dim = 256

model_simple = Sequential()
model_simple.add(Dense(8192, activation='relu', input_shape=(X_train.shape[1],)))
model_simple.add(Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model_simple.add(Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model_simple.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model_simple.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model_simple.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model_simple.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model_simple.add(Dense(encoding_dim, activation='relu'))
model_simple.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model_simple.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model_simple.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model_simple.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model_simple.add(Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model_simple.add(Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model_simple.add(Dense(8192, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model_simple.add(Dense(X_train.shape[1], activation='relu'))



model_simple.compile(optimizer=keras.optimizers.Adam(learning_rate=0.000001), loss='mse')

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history_simple = model_simple.fit(X_train, y_train, epochs=80, batch_size=32, validation_split=0.1, shuffle=True, callbacks=[early_stopping])

# Access the loss values from history_simple
training_loss = history_simple.history['loss']
validation_loss = history_simple.history['val_loss']

# Plot the training and validation loss
epochs = range(1, len(training_loss) + 1)
plt.plot(epochs, training_loss, 'b-', label='Training Loss')
plt.plot(epochs, validation_loss, 'r-', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss for Simple')
plt.legend()
plt.show()


# In[130]:


model_simple.summary()


# In[131]:


# Reconstruction Errors
X_train_pred_lstm = model_lstm.predict(X_train)
train_mae_loss_lstm = np.mean(np.abs(X_train_pred_lstm - X_train), axis=1)


# In[132]:


# Calculate the predicted values from the Simple Autoencoder
X_train_pred_simple = model_simple.predict(X_train)

# Reshape X_train_pred_simple to match the shape of X_train
X_train_pred_simple = np.reshape(X_train_pred_simple, (X_train_pred_simple.shape[0], X_train_pred_simple.shape[1], 1))


# In[133]:


# Calculate the mean absolute error
train_mae_loss_simple = np.mean(np.abs(X_train_pred_simple - X_train), axis=1) 


# In[134]:


# Calculate MSE
mse_lstm = np.mean(np.square(X_train_pred_lstm - X_train))
mse_simple = np.mean(np.square(X_train_pred_simple - X_train))


# In[135]:


# Print the MSE scores
print("MSE (LSTM Autoencoder):..", mse_lstm)
print("MSE (Simple Autoencoder):", mse_simple)


# In[136]:


# Plotting the training loss for both autoencoders
plt.plot(history_lstm.history['loss'], label='LSTM Autoencoder')
plt.plot(history_simple.history['loss'], label='Simple Autoencoder')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss Comparison')
plt.show()


# In[137]:


# Reconstruction Errors
X_test_pred_lstm = model_lstm.predict(X_test)
test_mae_loss_lstm = np.mean(np.abs(X_test_pred_lstm - X_test), axis=1)


# In[138]:


X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1])  # Reshape X_test

X_test_pred_simple = model_simple.predict(X_test)
test_mae_loss_simple = np.mean(np.abs(X_test_pred_simple - X_test_reshaped), axis=1)


# In[139]:


plt.hist(test_mae_loss_lstm, bins=50, label='LSTM Autoencoder')
plt.hist(test_mae_loss_simple, bins=50, label='Simple Autoencoder')
plt.xlabel('Test MAE loss')
plt.ylabel('Number of samples')
plt.legend()
plt.title('Test MAE Loss Comparison')
plt.show()


# In[140]:


# Plot histograms of train MAE loss for each autoencoder
plt.figure(figsize=(10, 6))
plt.hist(train_mae_loss_lstm, bins=50, label='LSTM Autoencoder', alpha=0.7)
plt.hist(train_mae_loss_simple, bins=50, label='Simple Autoencoder', alpha=0.7)
plt.xlabel('Train MAE loss')
plt.ylabel('Number of Samples')
plt.legend()
plt.title('Train MAE Loss Comparison')
plt.show()


# ## Set reconstruction error threshold

# In[156]:


# Set reconstruction error threshold
threshold = 5*  np.mean(train_mae_loss_lstm)
threshold_simple = 5 * np.mean(train_mae_loss_simple)
#threshold=1.5
#threshold_simple= 
print('Reconstruction error threshold for LSTM:.............',threshold)
print('Reconstruction error threshold for Simple Autoencoder', threshold_simple)


# In[157]:


# Compute test MAE loss for each autoencoder
X_test_pred_simple = model_simple.predict(X_test)
X_test_pred_simple = np.squeeze(X_test_pred_simple)  # Remove the extra dimension

# Reshape X_test to match the shape of X_test_pred_simple
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1])

test_mae_loss_simple = np.mean(np.abs(X_test_pred_simple - X_test_reshaped), axis=1)


# In[158]:


X_test_pred_lstm = model_lstm.predict(X_test)
test_mae_loss_lstm = np.mean(np.abs(X_test_pred_lstm - X_test), axis=1)


# In[159]:


# Plot histograms of test MAE loss for each autoencoder
plt.figure(figsize=(10, 6))
plt.hist(test_mae_loss_lstm, bins=50, label='LSTM Autoencoder', alpha=0.7)
plt.hist(test_mae_loss_simple, bins=50, label='Simple Autoencoder', alpha=0.7)
plt.xlabel('Test MAE loss')
plt.ylabel('Number of Samples')
plt.legend()
plt.title('Test MAE Loss Comparison')
plt.show()


# In[160]:


# Create anomaly DataFrame
anomaly_df = pd.DataFrame(test[TIME_STEPS:])
anomaly_df['loss'] = test_mae_loss_lstm
anomaly_df['threshold'] = threshold
anomaly_df['anomaly'] = anomaly_df['loss'] > anomaly_df['threshold']


# In[161]:


anomaly_df


# In[162]:


anomaly_df.shape #lstm


# In[163]:


# Plot test loss vs. threshold
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=anomaly_df['Date'], y=anomaly_df['loss'], name='Test loss'))
fig.add_trace(go.Scatter(x=anomaly_df['Date'], y=anomaly_df['threshold'], name='Threshold'))
fig.update_layout(showlegend=True, title='Test Loss vs. Threshold for LSTM Autoencoder')
fig.show()


# In[164]:


# Plot detected anomalies
anomalies_1 = anomaly_df.loc[anomaly_df['anomaly'] == True]


# In[165]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=anomaly_df['Date'], y=scaler.inverse_transform(anomaly_df['Close'].values.reshape(-1, 1)).flatten(), name='Close price'))
fig.add_trace(go.Scatter(x=anomalies_1['Date'], y=scaler.inverse_transform(anomalies_1['Close'].values.reshape(-1, 1)).flatten(), mode='markers', name='Anomaly'))
fig.update_layout(showlegend=True, title='Detected Anomalies for LSTM')
fig.show()


# In[166]:


# Create anomaly DataFrame for Simple Autoencoder
anomalies_simple = pd.DataFrame(test[TIME_STEPS:])
anomalies_simple['loss'] = test_mae_loss_simple
anomalies_simple['threshold'] = threshold_simple
anomalies_simple['anomaly'] = anomalies_simple['loss'] > anomalies_simple['threshold']


# In[167]:


anomalies_simple


# In[168]:


anomalies_simple.shape


# In[169]:


import plotly.graph_objects as go

fig_simple = go.Figure()
fig_simple.add_trace(go.Scatter(x=anomalies_simple['Date'], y=anomalies_simple['loss'], name='Test loss'))
fig_simple.add_trace(go.Scatter(x=anomalies_simple['Date'], y=anomalies_simple['threshold'], name='Threshold'))
fig_simple.update_layout(showlegend=True, title='Test Loss vs. Threshold for Simple Autoencoder')
fig_simple.show()


# In[170]:


# Plot detected anomalies for simple autoencoder
anomalies_2 = anomalies_simple.loc[anomalies_simple['anomaly'] == True]


# In[171]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=anomalies_simple['Date'], y=scaler.inverse_transform(anomalies_simple['Close'].values.reshape(-1, 1)).flatten(), name='Close price'))
fig.add_trace(go.Scatter(x=anomalies_2['Date'], y=scaler.inverse_transform(anomalies_2['Close'].values.reshape(-1, 1)).flatten(), mode='markers', name='Anomaly'))
fig.update_layout(showlegend=True, title='Detected Anomalies for Simple Autoencoder')
fig.show()


# In[175]:


from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Reconstruction Errors
X_test_pred_lstm = model_lstm.predict(X_test)
test_mae_loss_lstm = np.mean(np.abs(X_test_pred_lstm - X_test), axis=1)

# Reshape X_test to match X_test_pred_simple shape
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1])

X_test_pred_simple = model_simple.predict(X_test_reshaped)
test_mae_loss_simple = np.mean(np.abs(X_test_pred_simple - X_test_reshaped), axis=1)

# Set threshold for anomaly detection
threshold_lstm = 5 * np.mean(train_mae_loss_lstm)
threshold_simple = 5 * np.mean(train_mae_loss_simple)
#threshold_simple= 4

# Create anomaly labels based on threshold
anomaly_labels_lstm = test_mae_loss_lstm > threshold_lstm
anomaly_labels_simple = test_mae_loss_simple > threshold_simple

# Evaluate LSTM Autoencoder
f1_lstm = f1_score(anomaly_labels_lstm, anomaly_labels_lstm)
recall_lstm = recall_score(anomaly_labels_lstm, anomaly_labels_lstm)
precision_lstm = precision_score(anomaly_labels_lstm, anomaly_labels_lstm)
confusion_mtx_lstm = confusion_matrix(anomaly_labels_lstm, anomaly_labels_lstm)

# Evaluate Simple Autoencoder
f1_simple = f1_score(anomaly_labels_simple, anomaly_labels_simple)
recall_simple = recall_score(anomaly_labels_simple, anomaly_labels_simple)
precision_simple = precision_score(anomaly_labels_simple, anomaly_labels_simple)
confusion_mtx_simple = confusion_matrix(anomaly_labels_simple, anomaly_labels_simple)

# Print evaluation metrics
print("LSTM Autoencoder:")
print("F1 Score:", f1_lstm)
print("Recall:", recall_lstm)
print("Precision:", precision_lstm)
print("Confusion Matrix:\n", confusion_mtx_lstm)

print("\nSimple Autoencoder:")
print("F1 Score:", f1_simple)
print("Recall:", recall_simple)
print("Precision:", precision_simple)
print("Confusion Matrix:\n", confusion_mtx_simple)


# In[176]:


import matplotlib.pyplot as plt
import seaborn as sns

# Define class labels for the confusion matrices
class_labels = ['Non-Anomaly', 'Anomaly']

# Plot confusion matrix for LSTM Autoencoder
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mtx_lstm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix - LSTM Autoencoder")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Plot confusion matrix for Simple Autoencoder
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mtx_simple, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix - Simple Autoencoder")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


# In[177]:


import matplotlib.pyplot as plt

# LSTM Autoencoder
lstm_val_predictions = model_lstm.predict(X_test)
lstm_val_losses = np.mean(np.square(lstm_val_predictions - X_test), axis=1)

# Simple Autoencoder
simple_val_predictions = model_simple.predict(X_test)
# Reshape X_test to match the shape of simple_val_predictions
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1])
simple_val_losses = np.mean(np.square(simple_val_predictions - X_test_reshaped), axis=1)

# Plotting the distribution
plt.figure(figsize=(10, 6))
plt.hist(lstm_val_losses, bins=50, label='LSTM Autoencoder', alpha=0.5)
plt.hist(simple_val_losses, bins=50, label='Simple Autoencoder', alpha=0.5)
plt.xlabel('Reconstruction Loss')
plt.ylabel('Frequency')
plt.title('Reconstruction Loss Distribution')
plt.legend()
plt.show()


# In[ ]:




