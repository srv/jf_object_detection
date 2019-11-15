###########################################################################################
#                                                                                         #
# Evaluator class: Implements the most popular metrics for object detection               #
#                                                                                         #
# Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)                               #
#        SMT - Signal Multimedia and Telecommunications Lab                               #
#        COPPE - Universidade Federal do Rio de Janeiro                                   #
#        Last modification: Oct 9th 2018                                                 #
###########################################################################################

import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from BoundingBox import *
from BoundingBoxes import *
from curves_from_evals import *
from utils import *


class Curves:

    def GetCurves(self, results, zeros, ones, roc=1, repre=1, plot=True):

        results = sorted(results, key=lambda conf: conf[0], reverse=True)

        recall = []
        precision = []
        accuracy = []
        fallout = []

        for conf_thr in range(100):

            # confidence threshold in 0-1 range
            conf_thr = conf_thr/100

            # delete preds with confidence lower than confidence threshold
            delete = list()
            for idx, r in enumerate(results):
                if r[0] < conf_thr:
                    delete.append(idx)
            for index in sorted(delete, reverse=True):
                del results[index]

            #contar tps y fps
            total_tp = 0
            total_fp = 0
            for idx, r in enumerate(results):
                tp = r.count('TP')
                fp = r.count('FP')
                total_tp = total_tp + tp
                total_fp = total_fp + fp

            total_fn = ones-total_tp

            p = 0
            if total_tp+total_fp != 0:
                p = total_tp/(total_tp+total_fp)
            r = total_tp/ones
            precision.append(p)
            recall.append(r)

            if roc == 1:
                total_tn = zeros - total_fp
                f = total_fp/zeros
                a = (total_tp+total_tn)/(ones+zeros)
                fallout.append(f)
                accuracy.append(a)

        trade = []
        for i in range(len(recall)):
            t = (recall[i]+precision[i])/2
            trade.append(t)

        best_thr = trade.index(max(trade))
        best_recall = recall[best_thr]
        best_precision =precision[best_thr]

        print('best confidence threshold: ' + str(best_thr) + ' with:')
        print('recall: ' + str(best_recall))
        print('precision: ' + str(best_precision))

        if plot:
            if repre == 1:

                plt.plot(recall, precision, 'ro')
                plt.axis([0, 1, 0, 1])
                plt.title('Precision vs Recall')
                plt.xlabel('recall')
                plt.ylabel('precision')
                plt.show()

            if roc == 1:

                plt.plot(fallout, recall, 'ro')
                plt.axis([0, 1, 0, 1])
                plt.title('Fallout vs Recall')
                plt.xlabel('Fallout')
                plt.ylabel('Recall')
                plt.show()

        return best_thr, best_recall, best_precision
