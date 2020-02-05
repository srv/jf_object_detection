

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

    # python quantify.py --path_in "" --path_gt "" --wsize 25 --wover 0.5

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in', help='txt input directory.')
    parser.add_argument('--path_gt', default="", help='gt file path.')
    parser.add_argument('--wsize', default=8, help='window size')
    parser.add_argument('--wover', default=0.25, help='window overlap')
    parsed_args = parser.parse_args(sys.argv[1:])

    path_in = parsed_args.path_in
    path_gt = parsed_args.path_gt
    wsize = int(parsed_args.wsize)
    wover = float(parsed_args.wover)

    wover_ip = int(wsize-(wover*wsize))

    gt = pd.read_excel(io=path_gt)
    gt = gt.values

    predictions_list = list()
    sim_list = list()

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

    for idx in range(99):

        cthr = idx/100
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


        last_del = gt.shape[0] - quantifications.shape[0]
        gt2 = gt
        if last_del>0:
            gt2 = gt[:-last_del, :]

        last_del = quantifications.shape[0] - gt.shape[0]
        quantifications2 = quantifications
        if last_del>0:
            quantifications2 = quantifications[:-last_del, :]

        eq = np.equal(gt2,quantifications2)
        compare = np.logical_and(eq[:,0], eq[:,1], eq[:,2])

        similarity  = (np.count_nonzero(compare)/compare.shape[0])*100
        sim_list.append(similarity)

    print(sim_list)
    best_sim = max(sim_list)
    max_ind = max([i for i, x in enumerate(sim_list) if x == best_sim])

    print("max similarity: " + str(best_sim) + "at ctrh: " + str(max_ind))

    z = 1






main()
