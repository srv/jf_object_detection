import os
import re
import glob
import sys
import argparse


'''
call:
python cthr.py --path_in a/b --path_out c/d --cthr 0.4

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


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in', help='txt input directory.')
    parser.add_argument('--path_out', help='txt output directory.')
    parser.add_argument('--cthr', help='min conf threshold to delete prediction.')
    parsed_args = parser.parse_args(sys.argv[1:])

    dir_in = parsed_args.path_in
    dir_out = parsed_args.path_out
    cthr = float(parsed_args.cthr)

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)




    for file in os.listdir(dir_in):

        delete = list()

        if re.search("\.(txt)$", file):  # if the file is a txt
            file_path = os.path.join(dir_in, file)

            predictions = getPredictions(file_path)
            predictions = sorted(predictions, key=lambda conf: conf[1], reverse=True)

            for i, prediction in enumerate(predictions):
                 if prediction[1] < cthr:
                     break

            predictions = predictions[:i]

            file_out = os.path.join(dir_out, file)

            with open(file_out, 'w') as f:
                for prediction in predictions:
                    f.write(prediction[0] + " " +
                            str(prediction[1]) + " " +
                            str(int(prediction[2])) + " " +
                            str(int(prediction[3])) + " " +
                            str(int(prediction[4])) + " " +
                            str(int(prediction[5])) + "\n")

print('Successfully ctrh applied.')


main()
