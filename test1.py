import struct
from struct import unpack
import numpy as np


def unpack_drawing(file_handle):
    key_id, = unpack('Q', file_handle.read(8))
    country_code, = unpack('2s', file_handle.read(2))
    recognized, = unpack('b', file_handle.read(1))
    timestamp, = unpack('I', file_handle.read(4))
    n_strokes, = unpack('H', file_handle.read(2))
    image = []
    for i in range(n_strokes):
        n_points, = unpack('H', file_handle.read(2))
        fmt = str(n_points) + 'B'
        x = unpack(fmt, file_handle.read(n_points))
        y = unpack(fmt, file_handle.read(n_points))
        image.append((x, y))

    return {
        'key_id': key_id,
        'country_code': country_code,
        'recognized': recognized,
        'timestamp': timestamp,
        'image': image
    }


def unpack_drawings(filename):
    with open(filename, 'rb') as f:
        while True:
            try:
                yield unpack_drawing(f)
            except struct.error:
                break
c=0
data = None
for drawing in unpack_drawings('/content/drive/My Drive/autoencoder/full_binary_airplane.bin'):
    x = drawing['image']
    print(c)
    c=c+1
    for i in x:
      temp = np.array(i[0]).reshape(1,-1)
      temp1=np.array(i[1]).reshape(1,-1)
      temp = np.concatenate((temp,temp1),axis=1)
      if(temp.shape[1] > 1000):
        temp.reshape((1,1000))
      else:
        temp = np.hstack([temp, np.zeros([1, 1000-len(i[0]+i[1])])])
      if(data is None):  
        data=temp.copy()
      else:
        data = np.concatenate((data,temp))

np_data=[]
c = 0
for i in data:
  l=i.tolist()
  temp = []
  for j in l:
    for k in j:
      temp.append(k)
  if(len(temp)>1000):
    temp = temp[:1000]
    print(len(temp))
  else:
    while(len(temp) != 1000):
      temp.append(0)
  np_data.append(temp)



from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import pandas as pd

input_img= Input(shape=(1000,))
encoded = Dense(units=512, activation='relu')(input_img)
encoded = Dense(units=256, activation='relu')(encoded)
encoded = Dense(units=128, activation='relu')(encoded)
decoded = Dense(units=256, activation='relu')(encoded)
decoded = Dense(units=1000, activation='relu')(decoded)

autoencoder=Model(input_img, decoded)
encoder = Model(input_img, encoded)
autoencoder.compile(optimizer='adadelta', loss='mse', metrics=['accuracy'])
autoencoder.fit(data, data,epochs=300,batch_size=256)
encoded_imgs = encoder.predict(data[:10])
predicted = autoencoder.predict(data[:10])