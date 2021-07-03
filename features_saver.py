import datetime
import random  # feature versioning

import numpy as np

import input_data

t = datetime.datetime.now()
newdate = datetime.datetime.strftime(t, "%H_%M_%m_%d")
sr = random.SystemRandom()
version_number = sr.getrandbits(12)


def save_experiment_data():
    """
    saving experiment data is not storage efficient, but it is really useful when
    we only experiment with the model not the data.
    """
    wanted_words = "left,right,forward,backward,stop,go"
    speech_feature = "cgram"
    features = input_data.GetData(
        wanted_words=wanted_words, feature=speech_feature)
    # initialize dataset
    features.initialize()
    model_settings = features.model_settings

    input_size = np.ones(features.sample_rate)
    input_size = features.speech_features.cgram_(input_size, 16000)
    model_name = f"{speech_feature}_{input_size.shape[0]}x{input_size.shape[1]}_{wanted_words.replace(',', '_')}"
    print("generating training, validation, testing data.")
    print("input size: ", input_size.shape, model_name)
    x_train, y_train = features.get_data(-1, 0, 'training')
    np.save("data/x_train_{}.npy".format(model_name), x_train)
    np.save("data/y_train_{}.npy".format(model_name), y_train)

    x_val, y_val = features.get_data(-1, 0, 'validation')
    np.save("data/x_val_{}.npy".format(model_name), x_val)
    np.save("data/y_val_{}.npy".format(model_name), y_val)

    x_test, y_test = features.get_data(-1, 0, 'testing')
    np.save("data/x_test_{}.npy".format(model_name), x_test)
    np.save("data/y_test_{}.npy".format(model_name), y_test)


if __name__ == '__main__':
    save_experiment_data()
