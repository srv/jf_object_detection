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


class Evaluator:

    def GetPascalVOCMetrics(self,boundingboxes,conf_thr=0.7,IOUThreshold=0.5,method=MethodAveragePrecision.EveryPointInterpolation,showGraphic=True):

        print("-------------------------------------------------------")

        groundTruths = []  # gt list
        detections = []  # detection list
        classes = []

        # Loop through all bounding boxes and separate them into GTs and detections
        # [imageName, class, confidence, (bb coordinates XYX2Y2)]

        for bb in boundingboxes.getBoundingBoxes():

            if bb.getBBType() == BBType.GroundTruth:
                groundTruths.append([bb.getImageName(), bb.getClassId(), 1, bb.getAbsoluteBoundingBox(BBFormat.XYX2Y2)])
            else:
                detections.append([bb.getImageName(), bb.getClassId(), bb.getConfidence(), bb.getAbsoluteBoundingBox(BBFormat.XYX2Y2)])

            # get classes
            if bb.getClassId() not in classes:
                classes.append(bb.getClassId())
        classes = sorted(classes)
        n_classes = len(classes)

        count_gt = np.zeros(n_classes, dtype=int)
        conf_m = np.zeros([n_classes, n_classes + 1], dtype=int)

        dict = {}
        for idx, cls in enumerate(classes):
            dict[cls] = idx

        # delete predictions with confidence < cthr
        delete = list()
        for idx, d in enumerate(detections):
            if d[2] < conf_thr:
                delete.append(idx)
        for index in sorted(delete, reverse=True):
            del detections[index]



        for igt, gt in enumerate(groundTruths):

            n_c = dict[gt[1]]

            count_gt[n_c] = count_gt[n_c] + 1

            dets = [dets for dets in detections if dets[0] == gt[0]]

            iouMax = sys.float_info.min

            for dt in range(len(dets)):
                iou = Evaluator.iou(gt[3], dets[dt][3])
                if iou > iouMax:
                    iouMax = iou
                    dtmax = dt

            if iouMax >= IOUThreshold:
                if gt[1] == dets[dtmax][1]:
                    conf_m[n_c, n_c] = conf_m[n_c, n_c] + 1
                else:
                    n_pred = dict[dets[dtmax][1]]
                    conf_m[n_c, n_pred] = conf_m[n_c, n_pred] + 1

        sum_conf_m = conf_m.sum(axis=1)

        for c in range(n_classes):
            conf_m[c, n_classes] = count_gt[c] - sum_conf_m[c]

        print(conf_m)
        print(classes)


        return conf_m

    def PlotPrecisionRecallCurve(self,classId,boundingBoxes,conf_thr=0.7,IOUThreshold=0.5,method=MethodAveragePrecision.EveryPointInterpolation,
                                 showAP=False,showInterpolatedPrecision=False,savePath=None,showGraphic=True):

        results = self.GetPascalVOCMetrics(boundingBoxes, conf_thr, IOUThreshold, method, showGraphic)
        result = None
        for res in results:
            if res['class'] == classId:
                result = res
                break
        if result is None:
            raise IOError('Error: Class %d could not be found.' % classId)

        return result

    @staticmethod
    def CalculateAveragePrecision(rec, prec):

        mrec = []
        mrec.append(0)
        [mrec.append(e) for e in rec]
        mrec.append(1)

        mpre = []
        mpre.append(0)
        [mpre.append(e) for e in prec]
        mpre.append(0)

        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        ii = []
        for i in range(len(mrec) - 1):
            if mrec[1:][i] != mrec[0:-1][i]:
                ii.append(i + 1)
        ap = 0

        for i in ii:
            ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])

        return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]

    @staticmethod
    # 11-point interpolated average precision
    def ElevenPointInterpolatedAP(rec, prec):
        # def CalculateAveragePrecision2(rec, prec):
        mrec = []
        # mrec.append(0)
        [mrec.append(e) for e in rec]
        # mrec.append(1)
        mpre = []
        # mpre.append(0)
        [mpre.append(e) for e in prec]
        # mpre.append(0)
        recallValues = np.linspace(0, 1, 11)
        recallValues = list(recallValues[::-1])
        rhoInterp = []
        recallValid = []
        # For each recallValues (0, 0.1, 0.2, ... , 1)
        for r in recallValues:
            # Obtain all recall values higher or equal than r
            argGreaterRecalls = np.argwhere(mrec[:-1] >= r)
            pmax = 0
            # If there are recalls above r
            if argGreaterRecalls.size != 0:
                pmax = max(mpre[argGreaterRecalls.min():])
            recallValid.append(r)
            rhoInterp.append(pmax)
        # By definition AP = sum(max(precision whose recall is above r))/11
        ap = sum(rhoInterp) / 11
        # Generating values for the plot
        rvals = []
        rvals.append(recallValid[0])
        [rvals.append(e) for e in recallValid]
        rvals.append(0)
        pvals = []
        pvals.append(0)
        [pvals.append(e) for e in rhoInterp]
        pvals.append(0)
        # rhoInterp = rhoInterp[::-1]
        cc = []
        for i in range(len(rvals)):
            p = (rvals[i], pvals[i - 1])
            if p not in cc:
                cc.append(p)
            p = (rvals[i], pvals[i])
            if p not in cc:
                cc.append(p)
        recallValues = [i[0] for i in cc]
        rhoInterp = [i[1] for i in cc]
        return [ap, rhoInterp, recallValues, None]

    # For each detections, calculate IOU with reference
    @staticmethod
    def _getAllIOUs(reference, detections):
        ret = []
        bbReference = reference.getAbsoluteBoundingBox(BBFormat.XYX2Y2)
        # img = np.zeros((200,200,3), np.uint8)
        for d in detections:
            bb = d.getAbsoluteBoundingBox(BBFormat.XYX2Y2)
            iou = Evaluator.iou(bbReference, bb)
            # Show blank image with the bounding boxes
            # img = add_bb_into_image(img, d, color=(255,0,0), thickness=2, label=None)
            # img = add_bb_into_image(img, reference, color=(0,255,0), thickness=2, label=None)
            ret.append((iou, reference, d))  # iou, reference, detection
        # cv2.imshow("comparing",img)
        # cv2.waitKey(0)
        # cv2.destroyWindow("comparing")
        return sorted(ret, key=lambda i: i[0], reverse=True)  # sort by iou (from highest to lowest)

    @staticmethod
    def iou(boxA, boxB):
        # if boxes dont intersect
        if Evaluator._boxesIntersect(boxA, boxB) is False:
            return 0
        interArea = Evaluator._getIntersectionArea(boxA, boxB)
        union = Evaluator._getUnionAreas(boxA, boxB, interArea=interArea)
        # intersection over union
        iou = interArea / union
        assert iou >= 0
        return iou

    # boxA = (Ax1,Ay1,Ax2,Ay2)
    # boxB = (Bx1,By1,Bx2,By2)
    @staticmethod
    def _boxesIntersect(boxA, boxB):
        if boxA[0] > boxB[2]:
            return False  # boxA is right of boxB
        if boxB[0] > boxA[2]:
            return False  # boxA is left of boxB
        if boxA[3] < boxB[1]:
            return False  # boxA is above boxB
        if boxA[1] > boxB[3]:
            return False  # boxA is below boxB
        return True

    @staticmethod
    def _getIntersectionArea(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # intersection area
        return (xB - xA + 1) * (yB - yA + 1)

    @staticmethod
    def _getUnionAreas(boxA, boxB, interArea=None):
        area_A = Evaluator._getArea(boxA)
        area_B = Evaluator._getArea(boxB)
        if interArea is None:
            interArea = Evaluator._getIntersectionArea(boxA, boxB)
        return float(area_A + area_B - interArea)

    @staticmethod
    def _getArea(box):
        return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)