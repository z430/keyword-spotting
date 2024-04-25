def linear_model(input_shape, nclass):
    model = Sequential()
    model.add(Dense(nclass, input_shape=input_shape))
    return model
