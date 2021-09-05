import keras
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from scipy import interp
from sklearn.metrics import accuracy_score, auc, classification_report, roc_curve


def generate_results():
    models = [
        # 'data/results/cnn_cgram_81x20.npy',
        # 'data/results/cnn_mfccpsf_49x40_377_1357.npy',
        'data/results/cnn_mfccpsf_49x40_377_1357.npy'
    ]
    y_test = np.load('data/y_test_mfcc_49x40_left_right_forward_backward_stop_go.npy')
    label_index = '__silence__,__unknwown__,left,right,forward,backward,stop,go'.split(',')
    num_class = len(label_index)
    colors = ['red', 'blue']

    plt.figure()
    for model, color in zip(models, colors):
        y_score = np.load(model)

        clr = classification_report(y_test, y_score.round(), target_names=label_index)
        print(clr)

        print("accuracy: ", accuracy_score(y_test, y_score.round()) * 100)

        fpr = dict()
        tpr = dict()

        roc_auc = dict()
        print((np.array(y_test)).shape, (np.array(y_score)).shape)
        for i in range(num_class):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_class)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(num_class):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= num_class

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        print("FRR: ", np.average(1 - tpr["macro"]))
        print("FAR: ", np.average(fpr["micro"]))
        print(tpr["micro"])
        # Plot all ROC curves

        plt.plot(fpr["micro"], (1 - tpr["micro"]),
                label=f"average ROC curve (area = {roc_auc['micro']:.2f}",
                color=color, linewidth=2)
        plt.xlim([-0.01, 0.4])
        plt.ylim([-0.01, 0.33])
        plt.xlabel('False Alarm Rate')
        plt.ylabel('False Rejection Rate')
        plt.title('ROC')
        plt.legend(loc="lower right")
    plt.show()


def main():
    model = keras.models.load_model('models/cnn_trad_fpool3_mfcc_49x40_left_right_forward_backward_stop_go.hdf5')
    label_index = ['_silence_', '_unknown_', 'backward', 'forward', 'go', 'left', 'right', 'stop']
    model.summary()
    labels = []
    wavs = []
    pred_labels = []
    pre_l = []
    feature = 'mfcc'
    if feature == 'mfcc':
        test_data = np.load('data/x_test_mfcc_49x40_left_right_forward_backward_stop_go.npy')
        test_label = np.load('data/y_test_mfcc_49x40_left_right_forward_backward_stop_go.npy')
    elif feature == 'cgram':
        test_data = np.load('data/x_test_cgram_81x20_left_right_forward_backward_stop_go.npy')
        test_label = np.load('data/y_test_cgram_81x20_left_right_forward_backward_stop_go.npy')

    print(test_data.shape, test_label.shape)

    for wav, label in tqdm.tqdm(zip(test_data, test_label), total=len(test_label)):
        wav = np.expand_dims(wav, axis=0)
        prediction = model.predict(wav)
        pred_la = label_index[int(np.argmax(prediction))]
        # print(pred_la, label_index[np.argmax(label)])
        pred_labels.append(prediction)
        pre_l.append(pred_la)
        wavs.append(wav)
        labels.append(label)

    y_pred = np.array(pred_labels)
    if len(y_pred.shape) > 2:
        y_pred = np.reshape(y_pred, (y_pred.shape[0], y_pred.shape[-1]))
    print(y_pred.shape)
    print(y_pred.shape, test_label.shape)
    np.save('data/results/cnn_mfccpsf_49x40_377_1357.npy', y_pred)
    clr = classification_report(
        test_label, y_pred.round(), target_names=label_index)
    print(clr)


if __name__ == '__main__':
    main()
    generate_results()
