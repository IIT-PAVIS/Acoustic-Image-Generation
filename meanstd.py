import numpy as np
import glob
import pandas as pd

'''Compute meand and standard deviation for 5 runs od unet or dualcamnet choosing'''

def printmean(a, s, multiply=False):
    if multiply:
        a = [i * 100 for i in a]
    original = a.copy()
    minimum = np.min(a)
    maximum = np.max(a)
    a.remove(minimum)
    a.remove(maximum)
    a = np.array(a)
    print(a)
    mean = np.mean(a)
    std = np.std(a)
    print("{} {:.4f}+-{:.4f}".format(s, mean, std))
    original = ["{:.4f}".format(i) for i in original]
    original.append("{:.4f}+-{:.4f}".format(mean, std))
    return original

def computemean(a):
    b = a.copy()
    minimum = np.min(b)
    maximum = np.max(b)
    b.remove(minimum)
    b.remove(maximum)
    b = np.array(b)
    print(b)
    mean = np.mean(b)
    return mean

a = "/data/checkpointsaudiovideo/recdualcamnetunetacresnet/rec_Dualcamnet_2conn_*"
path = str.join('/', a.split('/')[:-1])
data_dirs = sorted(glob.glob(a))
mse =[]
iou = []
iouflickr = []
acc = []
accdualcam = []
area = []
areaflickr = []
classacc = []
unet = False
mfccmap = False
dualcamnetrec = True
areacompute = False
knn = False
for d in data_dirs:
    if unet:
        file = sorted(glob.glob("{}/UNet_testing_Acoustictry_*/intersection_0.5_accuracy.txt".format(d)))
        file = file[0]
        with open(file, "r") as outfile:
            t = outfile.read()
            iou1 = float(t.split(' ')[1])
            iou.append(iou1)

        file = sorted(glob.glob("{}/UNet_test_AcousticFrames_*/intersection_0.5_accuracy.txt".format(d)))
        file = file[0]
        with open(file, "r") as outfile:
            t = outfile.read()
            iouflickr1 = float(t.split(' ')[1])
            iouflickr.append(iouflickr1)
        accsub = []
        accdualcamsub = []
        file = sorted(glob.glob("{}/test_unet*_dualcamnet*.txt".format(d)))
        for f in file:
            with open(f, "r") as outfile:
                t = outfile.read()
                t2 = t.split(' acc ac')[0]
                accuracy1 = float(t2.split('acc rec ')[1])
                accdual1 = float(t.split(' acc ac')[1])
                accsub.append(accuracy1)
                accdualcamsub.append(accdual1)
        acc.append(computemean(accsub))
        accdualcam.append(computemean(accdualcamsub))

        file = sorted(glob.glob("{}/test_accuracy_*.txt".format(d)))
        file = file[0]
        with open(file, "r") as outfile:
            t = outfile.read()
            t = t.split('\t')[0]
            acc1 = float(t.split('Testing_Loss: ')[1])
            mse.append(acc1)

        if areacompute:
            file = sorted(glob.glob("{}/UNet_testing_Acoustictry_*/area.txt".format(d)))
            file = file[0]
            with open(file, "r") as outfile:
                t = outfile.read()
                area1 = float(t.split(' ')[1])
                area.append(area1)
            file = sorted(glob.glob("{}/UNet_test_AcousticFrames_*/area.txt".format(d)))
            file = file[0]
            with open(file, "r") as outfile:
                t = outfile.read()
                areaflickr1 = float(t.split(' ')[1])
                areaflickr.append(areaflickr1)
        if knn:
            file = sorted(glob.glob("{}/testing_Audio_*_testing_knn_value.txt".format(d)))
            file = file[0]
            with open(file, "r") as outfile:
                t = outfile.read()
                t = t.split(' ')[0]
                classacc1 = float(t.split('=')[1])
                classacc.append(classacc1)

    else:
        if dualcamnetrec:
            file = sorted(glob.glob("{}/test_unet*_dualcamnet*.txt".format(d)))
            file = file[0]
            with open(file, "r") as outfile:
                t = outfile.read()
                t2 = t.split(' acc ac')[0]
                accuracy1 = float(t2.split('acc rec ')[1])
                accdual1 = float(t.split(' acc ac')[1])
                acc.append(accuracy1)
                accdualcam.append(accdual1)
        else:
            if mfccmap:
                file = sorted(glob.glob("{}/test_accuracy_mfccmap_*.txt".format(d)))
            else:
                file = sorted(glob.glob("{}/test_accuracy.txt".format(d)))
            file = file[0]
            with open(file, "r") as outfile:
                t = outfile.read()
                t = t.split('\n')[1]
                acc1 = float(t.split('Testing_Accuracy: ')[1])
                accdualcam.append(acc1)

if unet:

    iou = printmean(iou, "iou")
    iouflickr = printmean(iouflickr, "iou flickr")
    acc = printmean(acc, "acc generated")
    accdualcam = printmean(accdualcam, "acc dualcamnet")
    mse = printmean(mse, "mse", True)
    dataset = pd.DataFrame({'iou': iou, 'acc':acc, 'mse':mse, 'iouflickr': iouflickr})
    if areacompute:
        area = printmean(area, "area")
        areaflickr = printmean(areaflickr, "area flickr")
        dataset['area'] = area
        dataset['areaflickr'] = areaflickr
    if knn:
        classacc = printmean(classacc, "knn accuracy")
        dataset['knnaccuracy'] = classacc
    print(dataset)
    dataset.to_excel(r'{}/export_dataframe2.xlsx'.format(path), index=False, header=True)
else:
    if dualcamnetrec:
        accdualcam = printmean(accdualcam, "acc_dualcam", False)
        acc = printmean(acc, "acc_generated", False)
        dataset = pd.DataFrame({'acc_dualcam': accdualcam, 'acc_gen': acc})
        dataset.to_excel(r'{}/export_dataframe.xlsx'.format(path), index=False, header=True)
    else:
        accdualcam = printmean(accdualcam, "acc_dualcam", False)
        dataset = pd.DataFrame({'acc': accdualcam})
        if mfccmap:
            dataset.to_excel(r'{}/export_dataframe_mfccmap.xlsx'.format(path), index=False, header=True)
        else:
            dataset.to_excel(r'{}/export_dataframe2.xlsx'.format(path), index=False, header=True)
