import os
import re
import glob
import sys
import argparse
from scipy import ndimage
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import scipy
from PIL import Image


'''
call:
python printbb.py --path_im a/b --path_txt e/f --path_out c/d

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


def getBoxFromPred(pred):
    box = (pred[2], pred[3], pred[4], pred[5])
    return box

def getColor(cls):

    if cls == "noctiluca":
        color = "thistle"

    if cls == "pulmo":
        color = "brown"

    if cls == "tuberculata":
        color = "coral"

    if cls == "achlyos":
        color = "cyan"

    if cls == "aurita":
        color = "gold"

    if cls == "branchi":
        color = "gray"

    if cls == "capilata":
        color = "green"

    if cls == "fuscesens":
        color = "greenyellow"

    if cls == "hysoscella":
        color = "olive"

    if cls == "lamarckii":
        color = "orange"

    if cls == "lutem":
        color = "red"

    if cls == "meleagris":
        color = "teal"

    if cls == "nemurai":
        color = "skyblue"

    if cls == "ohboya":
        color = "bisque"

    if cls == "quinquecirrha":
        color = "darkgreen"

    return color


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_im', help='im input directory.')
    parser.add_argument('--path_txt', help='txt input directory.')
    parser.add_argument('--path_out', help='im output directory.')
    parsed_args = parser.parse_args(sys.argv[1:])

    dir_im = parsed_args.path_im
    dir_txt = parsed_args.path_txt
    dir_out = parsed_args.path_out

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    for file in os.listdir(dir_txt):

        if re.search("\.(txt)$", file):  # if the file is a txt

            name, ext = file.split(".")
            path_im = os.path.join(dir_im, name + ".jpg")

            image = ndimage.imread(path_im, mode="RGB")  # read image

            image_pil = Image.fromarray(image)

            file_path = os.path.join(dir_txt, file)

            predictions = getPredictions(file_path)
            predictions = sorted(predictions, key=lambda conf: conf[1])

            for i, prediction in enumerate(predictions):
                box = getBoxFromPred(prediction)
                (left, top, right, bottom) = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                cls = prediction[0]
                conf = int(100*prediction[1])
                text = str(cls) + ' ' + str(conf) + "%"

                draw = ImageDraw.Draw(image_pil)

                color = getColor(cls)

                draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=3, fill=color)

                text_height = 12
                text_width = 90
                margin = 0

                # font = ImageFont.load_default()

                fontPath = "/home/miguel/Mango/jre7/lib/fonts/LucidaBrightDemiBold.ttf"
                font = ImageFont.truetype(fontPath, 10)



                if top > 80:
                    text_bottom = top
                else:
                    text_bottom = bottom + text_height

                draw.rectangle([(left, text_bottom - text_height), (left + text_width, text_bottom)], fill=color)
                draw.text((left + margin, text_bottom - text_height - margin), text, fill='black', font=font)

            save_path = os.path.join(dir_out, name + ".jpg")

            scipy.misc.imsave(save_path, image_pil)  # generate image file



main()
