from keras.layers import *
from keras.models import Sequential


def create_cnn_models(input_shape, nclass):
    print("input :, ", input_shape)
    model = Sequential()
    model.add(Reshape((49, 40, 1), input_shape=input_shape))
    model.add(Conv2D(filters=64, kernel_size=(20, 8), strides=(1, 1), activation='relu', input_shape=(98, 13, 1)))
    model.add(Dropout(0.5))
    model.add(MaxPool2D((1, 1), data_format='channels_first'))
    model.add(Conv2D(filters=32, kernel_size=(10, 4), strides=(1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(nclass, activation='sigmoid'))
    return model


def create_cnn_trad_fpool3(input_shape, nclass):
    print("input :, ", input_shape)
    model = Sequential()
    model.add(Reshape((1, 49, 40), input_shape=input_shape))
    model.add(Conv2D(filters=64, kernel_size=(20, 8), strides=(1, 1), activation='relu', input_shape=(98, 13, 1)))
    model.add(Dropout(0.5))
    model.add(MaxPool2D((1, 1), data_format='channels_first'))
    model.add(Conv2D(filters=64, kernel_size=(10, 4), strides=(1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(nclass, activation='sigmoid'))
    return model


def select_model(input_shape, nclass, model_name):
    if model_name == 'cnn':
        return create_cnn_models(input_shape, nclass)
    elif model_name == 'cnn_trad_fpool3':
        return create_cnn_models(input_shape, nclass)
