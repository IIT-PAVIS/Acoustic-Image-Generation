import argparse
from datetime import datetime
import numpy as np
import os
from scipy.spatial import distance
import matplotlib.pyplot as plt
import itertools

'''Retrieval not used now'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('init_checkpoint', type=str)
    parser.add_argument('anchor', type=str)
    parser.add_argument('gallery', type=str)
    parser.add_argument('set', type=str)
    parser.add_argument('datatype', type=str)
    parsed_args = parser.parse_args()

    init_checkpoint = parsed_args.init_checkpoint
    encoder_type = parsed_args.anchor
    encoder_type1 = parsed_args.gallery
    dataset = parsed_args.set
    s = init_checkpoint.split('/')[-1]
    namecheckpoint = (s.split('_')[1]).split('.ckpt')[0]
    path = str.join('/', init_checkpoint.split('/')[:-1])
    data_dir = '{}/{}_{}_{}'.format(path, dataset, encoder_type, namecheckpoint)
    data_dir1 = '{}/{}_{}_{}'.format(path, dataset, encoder_type1, namecheckpoint)
    if os.path.isfile('{}/{}_labels.npy'.format(data_dir, dataset)) \
            and os.path.isfile('{}/{}_data.npy'.format(data_dir, dataset)):
        features = np.load('{}/{}_data.npy'.format(data_dir, dataset))
        # features = np.reshape(features, (features.shape[0], -1))
        labels = np.load('{}/{}_labels.npy'.format(data_dir, dataset))
        scenario = np.load('{}/{}_scenario.npy'.format(data_dir, dataset))
        print(labels.shape[0])
    if os.path.isfile('{}/{}_labels.npy'.format(data_dir1, dataset)) \
            and os.path.isfile('{}/{}_data.npy'.format(data_dir1, dataset)):
        features1 = np.load('{}/{}_data.npy'.format(data_dir1, dataset))
        # features1 = np.reshape(features, (features.shape[0], -1))
        labels1 = np.load('{}/{}_labels.npy'.format(data_dir1, dataset))
        scenario1 = np.load('{}/{}_scenario.npy'.format(data_dir1, dataset))
        print(labels1.shape[0])
    labels = np.argmax(labels, axis=1)
    labels1 = np.argmax(labels1, axis=1)
    datatype = parsed_args.datatype
    if datatype == 'outdoor':
        numcl = 10
    elif datatype == 'music':
        numcl = 9
    else:
        numcl = 14

    confusion_matrix1 = np.zeros([numcl, numcl], dtype=float)
    confusion_matrix5 = np.zeros([numcl, numcl], dtype=float)
    confusion_matrix10 = np.zeros([numcl, numcl], dtype=float)
    # compute number of samples for each class
    num_samples_class = np.zeros([numcl], dtype=int)
    rank1 = 0
    rank2 = 0
    rank5 = 0
    rank10 = 0
    rank30 = 0
    for a in range(features.shape[0]):
        featuresanchor = np.expand_dims(features[a], axis=0)
        featuresgallery = features1
        distancearray = distance.cdist(featuresanchor, featuresgallery, 'euclidean')
        print('{} distance matrix {} {}'.format(datetime.now(), a, np.shape(distancearray)[1]))
        # for every acoustic feature vector find close one
        index = np.argsort(distancearray)
        index = np.squeeze(index)
        # order distances and take position
        # if they belong to same class
        if labels[a] == labels[index[0]]:
            rank1 += 1
            rank2 += 1
            rank5 += 1
            rank10 += 1
            rank30 += 1
        elif labels[a] in labels[index[[0, 1]]]:
            rank2 += 1
            rank5 += 1
            rank10 += 1
            rank30 += 1
        elif labels[a] in labels[index[:5]]:
            rank5 += 1
            rank10 += 1
            rank30 += 1
        elif labels[a] in labels[index[:10]]:
            rank10 += 1
            rank30 += 1
        elif labels[a] in labels[index[:30]]:
            rank30 += 1
        num_samples_class[labels[a]] += 1
        # add in position of predicted class
        confusion_matrix1[labels[a], labels[index[0]]] += 1
        for b in range(5):
            confusion_matrix5[labels[a], labels[index[b]]] += 1
            confusion_matrix10[labels[a], labels[index[b]]] += 1
        for b in range(5, 10):
            confusion_matrix10[labels[a], labels[index[b]]] += 1
    # divide each row for number of samples of that row
    confusion_matrix1 = confusion_matrix1 / num_samples_class.reshape(-1, 1)
    confusion_matrix5 = confusion_matrix5 / num_samples_class.reshape(-1, 1)
    confusion_matrix10 = confusion_matrix10 / num_samples_class.reshape(-1, 1)
    # divide for rank > 1
    confusion_matrix5 = confusion_matrix5 / 5.0
    confusion_matrix10 = confusion_matrix10 / 10.0
    print(confusion_matrix1)
    print(confusion_matrix5)
    print(confusion_matrix10)
    if datatype == 'outdoor':
        classes = ['Train', 'Boat', 'Drone', 'Fountain', 'Drill',
                   'Razor', 'Hair dryer', 'Vacuumcleaner', 'Cart', 'Traffic']
    elif datatype == 'music':
        classes = ['Clarinet', 'Trumpet silver', 'Double bass', 'Flute', 'Percussion',
                   'Saxophone', 'Trombone', 'Horn', 'Violin']
    else:
        classes = ['Clapping', 'Snapping fingers', 'Speaking', 'Whistling', 'Playing kendama', 'Clicking', 'Typing',
                   'Knocking', 'Hammering', 'Peanut breaking', 'Paper ripping', 'Plastic crumpling',
                   'Paper shaking',
                   'Stick dropping']
    cmap = plt.cm.Blues
    plt.imshow(confusion_matrix10, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = confusion_matrix10.max() / 2.
    for i, j in itertools.product(range(confusion_matrix10.shape[0]), range(confusion_matrix10.shape[1])):
        plt.text(j, i, format(confusion_matrix10[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confusion_matrix10[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(data_dir1 + '/confusion_matrix.png')

    accuracy = 1.0 * rank1 / np.shape(distancearray)[1]
    rank2 = 1.0 * rank2 / np.shape(distancearray)[1]
    rank5 = 1.0 * rank5 / np.shape(distancearray)[1]
    rank10 = 1.0 * rank10 / np.shape(distancearray)[1]
    rank30 = 1.0 * rank30 / np.shape(distancearray)[1]
    print('Accuracy {:6f} rank2 {:6f} rank5 {:6f} rank10 {:6f} rank30 {:6f}'.format(accuracy, rank2, rank5, rank10,
                                                                                    rank30))
    file = open('{}_{}_{}_{}_retrieval.txt'.format(data_dir, encoder_type, encoder_type1, dataset), 'w')
    file.write(
        'Accuracy {:6f} rank2 {:6f} rank5 {:6f} rank10 {:6f} rank30 {:6f}'.format(accuracy, rank2, rank5, rank10,
                                                                                  rank30))
    file.close()

if __name__ == '__main__':
    main()