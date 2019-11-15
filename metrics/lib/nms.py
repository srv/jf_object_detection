
import os
import re
import glob
import sys
import argparse


'''
call:
python nms.py --path_in a/b --path_out c/d --thr 0.4

'''

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
        x = float(splitLine[2])
        y = float(splitLine[3])
        w = float(splitLine[4])
        h = float(splitLine[5])
        pred = (idClass, confidence, x, y, w, h)

        predictions.append(pred)

    fh1.close()

    return predictions


def getIntersectionArea(boxA, boxB):
    if boxesIntersect(boxA, boxB) is False:
        return 0

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # intersection area
    return (xB - xA) * (yB - yA)


def boxesIntersect(boxA, boxB):
    if boxA[0] > boxB[2]:
        return False  # boxA is right of boxB
    if boxB[0] > boxA[2]:
        return False  # boxA is left of boxB
    if boxA[3] < boxB[1]:
        return False  # boxA is above boxB
    if boxA[1] > boxB[3]:
        return False  # boxA is below boxB
    return True


def getBoxArea(box):
    area = (box[2]-box[0])*(box[3]-box[1])
    return area


def getBoxFromPred(pred):
    box = (pred[2], pred[3], pred[4], pred[5])
    return box


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in', help='txt input directory.')
    parser.add_argument('--path_out', help='txt output directory.')
    parser.add_argument('--thr', help='min iou threshold to delete prediction.')
    parsed_args = parser.parse_args(sys.argv[1:])

    dir_in = parsed_args.path_in
    dir_out = parsed_args.path_out
    thr = float(parsed_args.thr)

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    for file in os.listdir(dir_in):
        #print(file)

        if re.search("\.(txt)$", file):  # if the file is a txt
            file_path = os.path.join(dir_in, file)

            predictions = getPredictions(file_path)
            predictions = sorted(predictions, key=lambda conf: conf[1], reverse=True)

            #print('predictions length: ' + str(len(predictions)))

            for i1, pred1 in enumerate(predictions):

                #print('hola, soy i1: ' + str(i1))
                #print('y ahora el predictions length es: ' + str(len(predictions)))

                intersectionArea1 = list()
                intersectionArea2 = list()
                delete = list()
                box1 = getBoxFromPred(pred1)
                area1 = getBoxArea(box1)

                for i2, pred2 in enumerate(predictions):

                    box2 = getBoxFromPred(pred2)
                    area2 = getBoxArea(box2)

                    inter = getIntersectionArea(box1, box2)
                    ioa1 = inter/area1
                    ioa2 = inter/area2

                    intersectionArea1.append(ioa1)
                    intersectionArea2.append(ioa2)

                for i3 in range(len(intersectionArea1)):

                    if (intersectionArea1[i3] > thr or intersectionArea2[i3] > thr) and i3 != i1:

                        delete.append(i3)

                for index in sorted(delete, reverse=True):
                    del predictions[index]

            file_out = os.path.join(dir_out, file)

            with open(file_out, 'w') as f:
                for prediction in predictions:
                    f.write(prediction[0] + " " +
                            str(prediction[1]) + " " +
                            str(int(prediction[2])) + " " +
                            str(int(prediction[3])) + " " +
                            str(int(prediction[4])) + " " +
                            str(int(prediction[5])) + "\n")

print('Successfully nms applied.')


main()

