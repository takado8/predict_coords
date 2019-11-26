import numpy as np
import tensorflow as tf
from tensorflow import keras
import datetime
import prepare_data
import os
import cv2


SIZE = 100

def newCNNNetwork(model_name):
    print(model_name)

    name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(name)
    tensBoard = keras.callbacks.TensorBoard(log_dir='log')

    img_rows, img_cols = SIZE, SIZE
    number_of_categories = 2
    x_train, y_train = prepare_data.load_data('train')

    if keras.backend.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        #x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        #x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train / 255

    # y_train = y_train / 128
    model = keras.models.Sequential()

    model.add(keras.layers.Conv2D(64, (2, 2), input_shape=input_shape))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.2))

    # model.add(keras.layers.Conv2D(128, (3, 3)))
    # model.add(keras.layers.Activation('relu'))
    # model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Flatten())
    #
    # model.add(keras.layers.Dense(512, activation='relu'))
    # model.add(keras.layers.Dropout(0.4))

    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(0.1))

    model.add(keras.layers.Dense(number_of_categories, activation='linear'))

    opt = keras.optimizers.Adam(decay=0.000000, lr=0.0015)
    model.compile(optimizer=opt, loss='mse', metrics=['mse'])
    #model.summary()
    model.fit(x_train, y_train, epochs=40, batch_size=32, shuffle=True, validation_split=0.0, callbacks=[tensBoard])
    model.save('model.ml')
    #sess.close()
    keras.backend.clear_session()


def predict():
    model = keras.models.load_model('model.ml')
    x_train,y_train = prepare_data.load_data('test')

    if keras.backend.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0],1,SIZE,SIZE)
        # x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1,SIZE,SIZE)
    else:
        x_train = x_train.reshape(x_train.shape[0],SIZE,SIZE,1)
        # x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (SIZE,SIZE,1)

    x_train = x_train / 255

    pred = []
    # for inp in x_train:
    pred.append(model.predict([x_train]))

    i=0
    for z in pred[0]:
        img = x_train[i] * 255
        cv2.circle(img,(int(z[0]*SIZE),int(z[1]*SIZE)),2,100,-1)
        img = cv2.resize(img,fx=10,fy=10, dsize=None)
        cv2.imwrite('data/test2/'+str(i)+'.jpg', img)
        print('prediction: ' + str(int(z[0]*SIZE)) + ',' + str(int(z[1]*SIZE)))
        print('actual: ' + str(y_train[i]*SIZE))
        i+=1


def model_summary(model_name):
    model = keras.models.load_model(os.path.join('models', model_name + '.ml'))
    model.summary()


