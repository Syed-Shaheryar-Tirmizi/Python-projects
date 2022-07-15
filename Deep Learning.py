import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Lambda, Flatten
from keras.optimizers import Adam,RMSprop
from keras.utils.np_utils import to_categorical
from keras.layers import BatchNormalization, Convolution2D, MaxPooling2D
from keras.preprocessing import image
from sklearn.model_selection import train_test_split

train = pd.read_csv("train.csv")
print(train.shape)
print(train.head())
test= pd.read_csv("test.csv")
print(test.shape)
test.head()


X_train = train.iloc[:,1:].values.astype('float32')
y_train = train.iloc[:,0].values.astype('int32')
X_test = test.values.astype('float32')


X_train = X_train.reshape(X_train.shape[0], 28, 28)
for i in range(6, 9):
    plt.subplot(330 + (i+1))
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
    plt.title(y_train[i])
    plt.show()


X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)
y_train= to_categorical(y_train)
num_classes = y_train.shape[1]
print(num_classes)


mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)

def standardize(x):
    return (x-mean_px)/std_px

plt.title(y_train[9])
plt.plot(y_train[9])
plt.xticks(range(10))
plt.show()


seed = 55
np.random.seed(seed)

model= Sequential()
model.add(Lambda(standardize,input_shape=(28,28,1)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
gen = image.ImageDataGenerator()


X = X_train
y = y_train
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)
batches = gen.flow(X_train, y_train, batch_size=64)
val_batches = gen.flow(X_val, y_val, batch_size=64)
history = model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=3,
                              validation_data=val_batches, validation_steps=val_batches.n)

def get_fc_model():
    model = Sequential([
        Lambda(standardize, input_shape=(28,28,1)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(10, activation='softmax')
        ])
    model.compile(optimizer='Adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

fc = get_fc_model()
fc.optimizer.lr = 0.01
history = fc.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1,
                    validation_data=val_batches, validation_steps=val_batches.n)

def get_cnn_model():
    model = Sequential([
        Lambda(standardize, input_shape=(28,28,1)),
        Convolution2D(32,(3,3), activation='relu'),
        Convolution2D(32,(3,3), activation='relu'),
        MaxPooling2D(),
        Convolution2D(64,(3,3), activation='relu'),
        Convolution2D(64,(3,3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(10, activation='softmax')
        ])
    model.compile(Adam(), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
model= get_cnn_model()
model.optimizer.lr=0.01
history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1,
                    validation_data=val_batches, validation_steps=val_batches.n)


def get_bn_model():
    model = Sequential([
        Lambda(standardize, input_shape=(28,28,1)),
        Convolution2D(32,(3,3), activation='relu'),
        BatchNormalization(axis=1),
        Convolution2D(32,(3,3), activation='relu'),
        MaxPooling2D(),
        BatchNormalization(axis=1),
        Convolution2D(64,(3,3), activation='relu'),
        BatchNormalization(axis=1),
        Convolution2D(64,(3,3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        BatchNormalization(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dense(10, activation='softmax')
        ])
    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model= get_bn_model()
model.optimizer.lr=0.01
history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1,
                    validation_data=val_batches, validation_steps=val_batches.n)

model.optimizer.lr=0.01
gen = image.ImageDataGenerator()
batches = gen.flow(X, y, batch_size=64)
history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=3)
predictions = model.predict_classes(X_test, verbose=0)

Prediction_dataframe = pd.DataFrame({"ImageID": list(range(1,len(predictions)+1)),
                         "Predicted": predictions})
print(Prediction_dataframe)