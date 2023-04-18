# Spam-SMS-Classification
## Dataset- 
The SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, tagged acording being ham (legitimate) or spam. \
Link: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
## Import Packages-
```
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt
import seaborn as sn
```
## Pre-processing-
```
df = pd.read_csv("spam.csv", encoding = "ISO-8859-1")
df.head()
df.groupby('category').describe()
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
df = df.rename(columns={'v1': 'category','v2':'message' })
df_balanced['spam'] = df_balanced['category'].map({"spam":1,"ham":0})
df_balanced['ham'] = df_balanced['category'].map({"spam":0,"ham":1})
```
## Fix Imbalanced dataset
```
df_spam = df[df['category'] =='spam']
df_spam.shape
df_ham = df[df['category'] =='ham']
df_ham.shape
df_ham_downsampled = df_ham.sample(df_spam.shape[0])
df_ham_downsampled.shape
df_balanced = pd.concat([df_spam, df_ham_downsampled])
```
## Data Split
```
X_train, X_test, y_train, y_test = train_test_split(df_balanced['message'], df_balanced['spam'], stratify = df_balanced['spam'])
```
## Used BERT NLP
```
bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
preprocessed_text = bert_preprocess(text_input)
outputs = bert_encoder(preprocessed_text)
```
## Model & Eval
```
l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])      #prevent overfitting
l = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(l)
model = tf.keras.Model(inputs=[text_input], outputs = [l])

METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall')
]

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=METRICS)
model.fit(X_train, y_train, epochs=10)

model.evaluate(X_test, y_test)
y_predicted = model.predict(X_test)
y_predicted = y_predicted.flatten()
y_predicted = np.where(y_predicted > 0.5, 1, 0)     #above 0.5 is spam returns 1 else 0
y_predicted
```
## confusion matrix representation using seaborn
![alt text](https://github.com/utkarshh27/Spam-SMS-Classification/blob/29a85cdf146a555f859a4305da3c437487dc5ad7/cm.png?raw=true)
```
cm = confusion_matrix(y_test, y_predicted)
cm 
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
```
## Custom Testing
```
reviews = [
    'Enter a chance to win $5000, hurry up, offer valid until march 31, 2021',
    'You are awarded a SiPix Digital Camera! call 09061221061 from landline. Delivery within 28days. T Cs Box177. M221BP. 2yr warranty. 150ppm. 16 . p pÂ£3.99',
    'it to 80488. Your 500 free text messages are valid until 31 December 2005.',
    'Hey Sam, Are you coming for a cricket game tomorrow',
    "Why don't you wait 'til at least wednesday to see if you get your ."
]
model.predict(reviews)
```
