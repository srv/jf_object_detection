

import os
import re
import sys
import argparse
import numpy as np
import scipy.misc
from skimage.util import *
from skimage.util import view_as_windows


def getPredictions(file):

    predictions = list()

    fh1 = open(file, "r")
    for line in fh1:
        line = line.replace("\n", "")
        if line.replace(' ', '') == '':
            continue
        splitLine = line.split(" ")

        idClass = (splitLine[0])  # class
        confidence = float(splitLine[1])
        pred = (idClass, confidence)

        predictions.append(pred)

    fh1.close()

    return predictions


def main():

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--path_in', help='txt input directory.')
    # parser.add_argument('--path_out', help='txt output directory.')
    # parser.add_argument('--cthr', default = 0, help='if eval 0, cthr to get quantification')
    # parser.add_argument('--print', default = 0, help='0 -> no print, 1 -> print quant (and gt)')
    # parser.add_argument('--wsize', default = 8, help='window size')
    # parser.add_argument('--wover', default = 0.25, help='window overlap')
    # parsed_args = parser.parse_args(sys.argv[1:])

    # path_in = parsed_args.path_in
    # path_out = parsed_args.path_out
    # cthr = float(parsed_args.cthr)
    # print = int(parsed_args.print)
    # wsize = int(parsed_args.wsize)
    # wover = float(parsed_args.wover)

    path_in = "test"
    # path_out = ""
    cthr = 0.5
    print = 1
    wsize = 8
    wover = 0.5
    wover = int(wover*wsize)

    predictions_list = list()

    # if not os.path.exists(path_out):
    #    os.makedirs(path_out)

    # read predictions
    for file in sorted(os.listdir(path_in)):

        if re.search("\.(txt)$", file):  # if the file is a txt

            file_path = os.path.join(path_in, file)

            predictions = getPredictions(file_path)
            predictions = sorted(predictions, key=lambda conf: conf[1], reverse=True)

            predictions_list.append(predictions)

    # list classes and create dict of unique classes
    classes = list()
    for i, predictions in enumerate(predictions_list):
        for j, pred in enumerate(predictions):
            classes.append(pred[0])
    u_classes = np.unique(classes)
    n_classes = len(u_classes)
    dict_classes = {}
    for i, name in enumerate(u_classes):
        dict_classes[u_classes[i]] = i

    # delete predicions with confidence < cthr
    predictions_cthr_list = list()

    for i, predictions in enumerate(predictions_list):
        for j, pred in enumerate(predictions):
            if pred[1] < cthr:
                predictions = predictions[:j]
                break
        predictions_cthr_list.append(predictions)

    # get number of isntances of each class for each image
    preds_count = np.zeros((len(predictions_list), n_classes), dtype=int)

    for i, predictions in enumerate(predictions_cthr_list):
        for j, pred in enumerate(predictions):
            c = pred[0]
            preds_count[i, dict_classes[c]] = preds_count[i, dict_classes[c]] + 1

    # apply windowing techniques
    n_counts = view_as_windows(np.choose(0, preds_count.T), 8, step=4).shape[0]
    preds_count_win = np.zeros((n_counts, n_classes), dtype=int)

    for i, name in enumerate(u_classes):
        counts = np.choose(i, preds_count.T)
        counts_win = view_as_windows(counts, wsize, step=wover)

        for j, win in enumerate(counts_win):
            values, rep = np.unique(win, return_counts=True)
            val = win[np.argmax(rep)]
            preds_count_win[j, i] = val

    # expand window quantification to all initial information points
    ip = n_counts*(wsize-wover) + wover
    quantifications = np.zeros((ip, n_classes), dtype=int)

    for i, name in enumerate(u_classes):
        counts_win = np.choose(i, preds_count_win.T)
        quant = np.repeat(counts_win, wsize-wover)
        for j in range(wover):
            quant = np.insert(quant, 0, quant[0])
        quantifications[..., i] = quant

    # save quantifications to csv

    if print == 1:
        z = 0
        # print

    z = 0









main()