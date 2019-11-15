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
    def GetPascalVOCMetrics(self,
                            boundingboxes,
                            conf_thr=0.7,
                            IOUThreshold=0.5,
                            method=MethodAveragePrecision.EveryPointInterpolation,
                            showGraphic=True):

        print("-------------------------------------------------------")

        """Get the metrics used by the VOC Pascal 2012 challenge.
        Get
        Args:
            boundingboxes: Object of the class BoundingBoxes representing ground truth and detected
            bounding boxes;
            IOUThreshold: IOU threshold indicating which detections will be considered TP or FP
            (default value = 0.5);
            method (default = EveryPointInterpolation): It can be calculated as the implementation
            in the official PASCAL VOC toolkit (EveryPointInterpolation), or applying the 11-point
            interpolatio as described in the paper "The PASCAL Visual Object Classes(VOC) Challenge"
            or EveryPointInterpolation"  (ElevenPointInterpolation);
        Returns:
            A list of dictionaries. Each dictionary contains information and metrics of each class.
            The keys of each dictionary are:
            dict['class']: class representing the current dictionary;
            dict['precision']: array with the precision values;
            dict['recall']: array with the recall values;
            dict['AP']: average precision;
            dict['interpolated precision']: interpolated precision values;
            dict['interpolated recall']: interpolated recall values;
            dict['total positives']: total number of ground truth positives;
            dict['total TP']: total number of True Positive detections;
            dict['total FP']: total number of False Negative detections;
        """
        ret = []  # list containing metrics (precision, recall, average precision) of each class
        # List with all ground truths (Ex: [imageName,class,confidence=1, (bb coordinates XYX2Y2)])
        general = [] # list containing general metrics
        groundTruths = []
        # List with all detections (Ex: [imageName,class,confidence,(bb coordinates XYX2Y2)])
        detections = []
        # Get all classes
        classes = []
        # Loop through all bounding boxes and separate them into GTs and detections
        for bb in boundingboxes.getBoundingBoxes():

            # [imageName, class, confidence, (bb coordinates XYX2Y2)]
            if bb.getBBType() == BBType.GroundTruth:
                groundTruths.append([
                    bb.getImageName(),
                    bb.getClassId(), 1,
                    bb.getAbsoluteBoundingBox(BBFormat.XYX2Y2)
                ])
            else:
                detections.append([
                    bb.getImageName(),
                    bb.getClassId(),
                    bb.getConfidence(),
                    bb.getAbsoluteBoundingBox(BBFormat.XYX2Y2)
                ])

            # get class
            if bb.getClassId() not in classes:
                classes.append(bb.getClassId())
        classes = sorted(classes)

        conf_hist_tp = list()
        conf_hist_fp = list()
        conf_hist = list()
        for idx, d in enumerate(detections):
            conf_hist.append(d[2]*100)

        bins = np.arange(0, 101, 0.5)  # fixed bin size
        plt.hist(conf_hist, bins=bins)
        plt.title('confidence histogram')
        plt.xlabel('confidende')
        plt.ylabel('count')
        if showGraphic is True:
            plt.show()


        delete = list()
        for idx, d in enumerate(detections):
            if d[2] < conf_thr:
                delete.append(idx)
        for index in sorted(delete, reverse=True):
            del detections[index]

        iou_hist = list()
        iou_hist2 = list()

        total_npos  = 0
        confidences  = np.array([])
        tpfp = np.array([])

        # Precision x Recall is obtained individually by each class
        # Loop through by classes


        # -------------------------------------
        print('len dects is: ' + str(len(detections)))
        for d in range(len(detections)):
            dects2 = [dects2 for dects2 in detections if dects2[0] == detections[d][0]]
            iouMax2 = sys.float_info.min
            for j in range(len(dects2)):
                iou2 = Evaluator.iou(detections[d][3], dects2[j][3])
                if iou2 < 1:
                    iou_hist2.append(iou2 * 100)
        # -------------------------------------


        for c in classes:

            print("class:" + c)


            # Get only detection of class c
            dects = []
            [dects.append(d) for d in detections if d[1] == c]
            # Get only ground truths of class c
            gts = []
            [gts.append(g) for g in groundTruths if g[1] == c]
            npos = len(gts)
            total_npos = total_npos + npos
            ndects = len(dects)
            print("no gts: " + str(npos))
            print("no dects: " + str(ndects))

            # sort detections by decreasing confidence
            dects = sorted(dects, key=lambda conf: conf[2], reverse=True)
            TP = np.zeros(len(dects))
            FP = np.zeros(len(dects))

            for d in range(len(dects)):
                confidences = np.append(confidences, dects[d][2])

            # create dictionary with amount of gts for each image
            det = Counter([cc[0] for cc in gts])
            for key, val in det.items():
                det[key] = np.zeros(val)

            # Loop through detections
            for d in range(len(dects)):

                # Find ground truth image
                gt = [gt for gt in gts if gt[0] == dects[d][0]]
                iouMax = sys.float_info.min

                for j in range(len(gt)):
                    iou = Evaluator.iou(dects[d][3], gt[j][3])
                    if iou > iouMax:
                        iouMax = iou
                        jmax = j

                iou_hist.append(iouMax*100)

                # Assign detection as true positive/don't care/false positive
                if iouMax >= IOUThreshold:
                    if det[dects[d][0]][jmax] == 0:
                        TP[d] = 1  # count as true positive
                        conf_hist_tp.append(dects[d][2] * 100)
                        tpfp = np.append(tpfp, 'TP')
                        det[dects[d][0]][jmax] = 1  # flag as already 'seen'
                    else:
                        FP[d] = 1  # count as false positive
                        conf_hist_fp.append(dects[d][2] * 100)
                        tpfp = np.append(tpfp, 'FP')

                # - A detected "cat" is overlaped with a GT "cat" with IOU >= IOUThreshold.
                else:
                    FP[d] = 1  # count as false positive
                    conf_hist_fp.append(dects[d][2] * 100)
                    tpfp = np.append(tpfp, 'FP')

            # compute precision, recall and average precision
            acc_FP = np.cumsum(FP)
            acc_TP = np.cumsum(TP)
            rec = acc_TP / npos
            prec = np.divide(acc_TP, (acc_FP + acc_TP))

            tps = np.count_nonzero(TP == 1.0)
            fps = np.count_nonzero(FP == 1.0)
            fns = len(gts) - tps
            print("tps: " + str(tps))
            print("fps: " + str(fps))
            print("fns: " + str(fns))

            # Depending on the method, call the right implementation
            if method == MethodAveragePrecision.EveryPointInterpolation:
                [ap, mpre, mrec, ii] = Evaluator.CalculateAveragePrecision(rec, prec)
            else:
                [ap, mpre, mrec, _] = Evaluator.ElevenPointInterpolatedAP(rec, prec)
            # add class result in the dictionary to be returned


            r = {
                'class': c,
                'precision': mpre,
                'recall': mrec,
                'AP': ap,
                'interpolated precision': mpre,
                'interpolated recall': mrec,
                'total positives': npos,
                'total detections': ndects,
                'total TP': np.sum(TP),
                'total FP': np.sum(FP),
                'total FN': fns
            }
            ret.append(r)



        res = []
        for a,b in enumerate(confidences):
            res.append([b, tpfp[a]])

        curves = Curves()
        best_thr, best_rec, best_prec = curves.GetCurves(res, zeros=0, ones=total_npos, roc=0, repre=1, plot=showGraphic)

        # print(iou_hist between preds)
        bins = np.arange(0, 101, 0.5)
        plt.hist(iou_hist2, bins=bins)
        plt.title('iou histogram between preds')
        plt.xlabel('iou')
        plt.ylabel('count')
        if showGraphic is True:
            plt.show()
        print('iou_hist2 length: ' + str(len(iou_hist2)))

        # print(iou_hist)
        bins = np.arange(0, 101, 0.5)  # fixed bin size
        plt.hist(conf_hist_tp, bins=bins, alpha=0.5)
        plt.hist(conf_hist_fp, bins=bins)
        plt.title('confidence TP and FP histogram')
        plt.xlabel('conf')
        plt.ylabel('count')
        if showGraphic is True:
            plt.show()
        print('conf_hist_fp length: ' + str(len(conf_hist_fp)))
        print('conf_hist_tp length: ' + str(len(conf_hist_tp)))

        # print(iou_hist)
        bins = np.arange(0, 101, 0.5)  # fixed bin size
        plt.hist(conf_hist_fp, bins=bins)
        plt.title('confidence FP histogram')
        plt.xlabel('conf')
        plt.ylabel('count')
        if showGraphic is True:
            plt.show()

        # print(iou_hist)
        bins = np.arange(0, 101, 0.5)  # fixed bin size
        plt.hist(iou_hist, bins=bins)
        plt.title('iou histogram')
        plt.xlabel('iou')
        plt.ylabel('count')
        if showGraphic is True:
            plt.show()
        print('iou_hist length: ' + str(len(iou_hist)))



        map = 0
        for i,c in enumerate(classes):
            map = map + ret[i]['AP']
        map = map / int(len(classes))
        print("map:" + str(round((map * 100), 2)) + "%")

        general = {
            'map': map*100,
            'conf_thr': best_thr,
            'recall': best_rec*100,
            'precision': best_prec*100
        }

        return ret, general

    def PlotPrecisionRecallCurve(self,
                                 classId,
                                 boundingBoxes,
                                 conf_thr=0.7,
                                 IOUThreshold=0.5,
                                 method=MethodAveragePrecision.EveryPointInterpolation,
                                 showAP=False,
                                 showInterpolatedPrecision=False,
                                 savePath=None,
                                 showGraphic=True):
        """PlotPrecisionRecallCurve
        Plot the Precision x Recall curve for a given class.
        Args:
            classId: The class that will be plot;
            boundingBoxes: Object of the class BoundingBoxes representing ground truth and detected
            bounding boxes;
            IOUThreshold (optional): IOU threshold indicating which detections will be considered
            TP or FP (default value = 0.5);
            method (default = EveryPointInterpolation): It can be calculated as the implementation
            in the official PASCAL VOC toolkit (EveryPointInterpolation), or applying the 11-point
            interpolatio as described in the paper "The PASCAL Visual Object Classes(VOC) Challenge"
            or EveryPointInterpolation"  (ElevenPointInterpolation).
            showAP (optional): if True, the average precision value will be shown in the title of
            the graph (default = False);
            showInterpolatedPrecision (optional): if True, it will show in the plot the interpolated
             precision (default = False);
            savePath (optional): if informed, the plot will be saved as an image in this path
            (ex: /home/mywork/ap.png) (default = None);
            showGraphic (optional): if True, the plot will be shown (default = True)
        Returns:
            A dictionary containing information and metric about the class. The keys of the
            dictionary are:
            dict['class']: class representing the current dictionary;
            dict['precision']: array with the precision values;
            dict['recall']: array with the recall values;
            dict['AP']: average precision;
            dict['interpolated precision']: interpolated precision values;
            dict['interpolated recall']: interpolated recall values;
            dict['total positives']: total number of ground truth positives;
            dict['total TP']: total number of True Positive detections;
            dict['total FP']: total number of False Negative detections;
        """
        results = self.GetPascalVOCMetrics(boundingBoxes, conf_thr, IOUThreshold, method, showGraphic)
        result = None
        for res in results:
            if res['class'] == classId:
                result = res
                break
        if result is None:
            raise IOError('Error: Class %d could not be found.' % classId)

        precision = result['precision']
        recall = result['recall']
        average_precision = result['AP']
        mpre = result['interpolated precision']
        mrec = result['interpolated recall']
        npos = result['total positives']
        total_tp = result['total TP']
        total_fp = result['total FP']

        c = result['class']




        if showInterpolatedPrecision:
            if method == MethodAveragePrecision.EveryPointInterpolation:
                plt.plot(mrec, mpre, '--r', label='Interpolated precision (every point)')
            elif method == MethodAveragePrecision.ElevenPointInterpolation:
                # Uncomment the line below if you want to plot the area
                plt.plot(mrec, mpre, 'or', label='11-point interpolated precision')
                # Remove duplicates, getting only the highest precision of each recall value
                nrec = []
                nprec = []
                for idx in range(len(mrec)):
                    r = mrec[idx]
                    if r not in nrec:
                        idxEq = np.argwhere(mrec == r)
                        nrec.append(r)
                        nprec.append(max([mpre[int(id)] for id in idxEq]))
                plt.plot(nrec, nprec, 'or', label='11-point interpolated precision')


        plt.plot(recall, precision, label='Precision')
        plt.xlabel('recall')
        plt.ylabel('precision')
        if showAP:
            ap_str = "{0:.2f}%".format(average_precision * 100)
            plt.title('Precision x Recall curve \nClass: %s, AP: %s' % (str(classId), ap_str))
            # plt.title('Precision x Recall curve \nClass: %s, AP: %.4f' % (str(classId),
            # average_precision))
        else:
            plt.title('Precision x Recall curve \nClass: %d' % classId)
        plt.legend(shadow=True)
        plt.grid()

        if savePath is not None:
            plt.savefig(savePath)
        if showGraphic is True:
            plt.show()
            # plt.waitforbuttonpress()
        ret = {}
        ret['class'] = classId
        ret['precision'] = precision
        ret['recall'] = recall
        ret['AP'] = average_precision
        ret['interpolated precision'] = mpre
        ret['interpolated recall'] = mrec
        ret['total positives'] = npos
        ret['total TP'] = total_tp
        ret['total FP'] = total_fp
        return ret

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