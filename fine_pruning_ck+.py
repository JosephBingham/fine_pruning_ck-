import tensorflow as tf
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
import numpy as np
import os
from os import walk
from PIL import Image as Im

######
# Get and format data

data = {}
ENCODE_EMOTION = {'anger':0, 'fear':1, "happy":2, "sadness":3, "surprise":4}

for i in os.listdir():
    if os.path.isdir(i) and not '.' in i:
        directory, _, filenames = next(walk(i))
        for f in filenames:
            individual = f.split('_')[0]
            Y = np.array([0]*5)
            Y[ENCODE_EMOTION[directory]] = 1
            if not individual in data:
                data[individual] = [(np.array(Im.open(i+ "/"+f)).reshape((48,48,1))/255., Y)]
            else:
                data[individual].append((np.array(Im.open(i+ "/"+f)).reshape((48,48,1))/255., Y))

test_ind = []
for k in data.keys():

    if len(data[k]) >= 12: # make sure individual has enough support 
        test_ind.append(k)

######




def make_model():
    # VGG19
    cnn_model = Sequential([
    Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=(48,48,1), padding = "same"),
    Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding = "same"), 
    MaxPooling2D(pool_size=2),
    Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding = "same"),
    Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding = "same"),
    MaxPooling2D(pool_size=2),
    Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding = "same"),
    Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding = "same"),
    Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding = "same"),
    Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding = "same"),
    MaxPooling2D(pool_size=2),
    Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding = "same"),
    Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding = "same"),
    Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding = "same"),
    Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding = "same"), 
    MaxPooling2D(pool_size=2),
    Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding = "same"),
    Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding = "same"),
    Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding = "same"),
    Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding = "same"),
    MaxPooling2D(pool_size=3),
    Flatten(),
    Dense(4096, activation='relu'),
    Dense(4096, activation='relu'),
    Dense(5, activation='softmax')
    ])
    
    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy')
    return cnn_model

import tensorflow_model_optimization as tfm
import tempfile

batch_size = 64

plm = tfm.sparsity.keras.prune_low_magnitude
end_step = np.ceil(10/batch_size).astype(np.int32)*20

pruning_params = {'pruning_schedule': tfm.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.70,
                                                               begin_step=0,
                                                               end_step=end_step)}

def prune(file, x, y):
    cnn_copy = keras.models.load_model(file)
    cnn_copy.fit(x, y, batch_size=batch_size, epochs=10, verbose=0)
    cnn_copy.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=['accuracy'])
    s2 = cnn_copy.evaluate(x, y)
    del(cnn_copy)
    
    cnn_model = keras.models.load_model(file)
    model_for_pruning = plm(cnn_model, **pruning_params)

    model_for_pruning.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=['accuracy'])

    logdir = tempfile.mkdtemp()

    callbacks = [tfm.sparsity.keras.UpdatePruningStep(), tfm.sparsity.keras.PruningSummaries(log_dir=logdir)]
    model_for_pruning.fit(x, y, batch_size=batch_size, epochs=5, callbacks=callbacks, verbose=0)
    s1 = model_for_pruning.evaluate(x, y)
    return (s1, s2)
    

accuracies = []
for ind in test_ind: #for each individual
    #init vars
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    for k in data.keys():
        if ind == k:
            for dat in data[k]:
                X_test.append(dat[0])
                Y_test.append(dat[1])
        else:
            for dat in data[k]:
                X_train.append(dat[0])
                Y_train.append(dat[1])
    

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    
    model = make_model()
    model.fit(X_train, Y_train, epochs=50, batch_size=batch_size, validation_split=1/10)
    model.save("source_" + ind + ".h5")
    accuracies.append(prune("source_" + ind + ".h5", X_test, Y_test))
    print(i, accuracies[-1])(base)
