

import os
import re
import matplotlib.pyplot as plt
import sys
import argparse
import numpy as np
# import scipy.misc
from skimage.util import *
from skimage.util import view_as_windows
from natsort import natsorted
import pandas as pd


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

    # python quantify.py --path_in "test" --path_out "" --cthr 0.5 --print_opt 1  --wsize 8 --wover 0.5

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in', help='txt input directory.')
    parser.add_argument('--path_out', default="", help='txt output directory.')
    parser.add_argument('--cthr', default=0, help='if eval 0, cthr to get quantification')
    parser.add_argument('--print_opt', default=0, help='0 -> no print, 1 -> print quant (and gt)')
    parser.add_argument('--wsize', default=8, help='window size')
    parser.add_argument('--wover', default=0.25, help='window overlap')
    parsed_args = parser.parse_args(sys.argv[1:])

    path_in = parsed_args.path_in
    path_out = parsed_args.path_out
    cthr = float(parsed_args.cthr)
    print_opt = int(parsed_args.print_opt)
    wsize = int(parsed_args.wsize)
    wover = float(parsed_args.wover)

    wover_ip = int(wsize-(wover*wsize))

    predictions_list = list()

    if path_out != "":
        if not os.path.exists(path_out):
            os.makedirs(path_out)

    # read predictions
    for file in natsorted(os.listdir(path_in)):

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
    n_counts = view_as_windows(np.choose(0, preds_count.T), wsize, step=wover_ip).shape[0]
    preds_count_win = np.zeros((n_counts, n_classes), dtype=int)

    for i, name in enumerate(u_classes):
        counts = np.choose(i, preds_count.T)
        counts_win = view_as_windows(counts, wsize, step=wover_ip)

        for j, win in enumerate(counts_win):
            values, rep = np.unique(win, return_counts=True)
            rep = np.flip(rep, 0)            # flip to give priority go higher number, as it is
            values = np.flip(values, 0)      # more usual that the network has FN rather than FP
            val = values[np.argmax(rep, 0)]  # and argmax chooses the first element in case of tie
            preds_count_win[j, i] = val

    # expand window quantification to all initial information points
    ip = n_counts*wover_ip+(wsize-wover_ip)
    quantifications = np.zeros((ip, n_classes), dtype=int)

    for i, name in enumerate(u_classes):
        counts_win = np.choose(i, preds_count_win.T)
        quant = np.repeat(counts_win, wover_ip)
        for j in range(wsize-wover_ip):
            quant = np.insert(quant, 0, quant[0])
        quantifications[..., i] = quant

    path_save_csv = os.path.join(path_out, "quant"+"_"+str(wsize)+"_"+str(wover)+"_"+str(cthr)+"_"+".csv")
    np.savetxt(path_save_csv, quantifications)  # save quantifications to csv

    if print_opt == 1:

        fig = plt.figure()
        plt.xlabel('information points')
        plt.ylabel('jellyfish counts')
        x = np.arange(0, quantifications.shape[0] , 1, dtype=int)
        for i, name in enumerate(u_classes):
            plt.plot(x, quantifications[..., i], label=name)

        plt.legend(loc='upper left', frameon=False)

        path_save_plot = os.path.join(path_out, "quant"+"_"+str(wsize)+"_"+str(wover)+"_"+str(cthr)+"_"+".png")
        plt.savefig(path_save_plot)  # save quantifications to plot


main()
