import numpy as np
import csv
import pickle

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.optimizers import Adam

number_of_classes = 7
dimension = 48
number_of_channels = 1

def load_dataset(file):
    dataset_features = []
    dataset_labels = []

    file = '/input' + file

    with open(file) as csvfile:
        csv_reader_object = csv.reader(csvfile, delimiter=',')
        count = 0
        for row in csv_reader_object:
            if len(row) == 0:
                _0 = 0  # ignore
            else:
                dataset_features.append(row[1].split())
                # print(count)
                # count += 1
                temp = np.zeros(number_of_classes, dtype=int)
                temp[row[0]] = int(1)
                dataset_labels.append(temp)

    return np.array(dataset_features), np.array(dataset_labels)

def pickle_dump(what, name):
    pickle_out = open(name, 'wb')
    pickle.dump(what, pickle_out)
    pickle_out.close()

def pickle_retrieve(name):
    pickle_in = open(name, 'rb')
    file = pickle.load(pickle_in)
    return file

dataset_train_features, dataset_train_labels = load_dataset('training.csv') # paste your training.csv file in the directory
dataset_test_features, dataset_test_labels = load_dataset('test.csv')   # paste your training.csv file in the directory
print(dataset_train_features.shape)

pickle_dump(dataset_train_features, '/input/dataset_train_features.pickle')
pickle_dump(dataset_train_labels, '/input/dataset_train_labels.pickle')
pickle_dump(dataset_test_features, '/input/dataset_test_features.pickle')
pickle_dump(dataset_test_labels, '/input/dataset_test_labels.pickle')

dataset_train_features = pickle_retrieve('/input/dataset_train_features.pickle')
dataset_train_labels = pickle_retrieve('/input/dataset_train_labels.pickle')
dataset_test_features = pickle_retrieve('/input/dataset_test_features.pickle')
dataset_test_labels = pickle_retrieve('/input/dataset_test_labels.pickle')

dataset_train_features = dataset_train_features.astype('float32')
dataset_test_features = dataset_test_features.astype('float32')
dataset_train_features = dataset_train_features / 255.0
dataset_test_features = dataset_test_features / 255.0

print('dataset_train_features.shape:', dataset_train_features.shape)
print('dataset_train_labels.shape:', dataset_train_labels.shape)
print('dataset_test_features.shape:', dataset_test_features.shape)
print('dataset_test_labels.shape:', dataset_test_labels.shape)

dataset_train_features = dataset_train_features.reshape((-1, 48, 48, 1))
dataset_test_features = dataset_test_features.reshape((-1, 48, 48, 1))

# model similar to VGG, with one block of Convolution-Convolution-Pooling and fully connedtec layer omitted. Also added dropout for robustness
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(48, 48 ,1), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(number_of_classes, activation='softmax'))

# Compile model
epochs = 300
lrate = 0.01
decay = lrate/epochs
adam = Adam(decay=decay)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
print(model.summary())
model.fit(dataset_train_features, dataset_train_labels, validation_data=(dataset_test_features, dataset_test_labels), epochs=epochs, batch_size=50)
# Final evaluation of the model
#scores = model.evaluate(dataset_test_features, dataset_test_labels, verbose=0)
#print("Accuracy: %.2f%%" % (scores[1]*100))

