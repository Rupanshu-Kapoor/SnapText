# import the necessary packages
from pytesseract import Output
import pytesseract
import argparse
import cv2
import os
from config import *
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# get the current working directory
cwd = os.getcwd()

# get the path to the input folder and list all the image files
images_path = os.path.join(cwd, input_folder)
image_names = os.listdir(images_path)

# create the output directory if it doesn't exist
output_dir = os.path.join(cwd, preprocessed_directory)
os.makedirs(output_dir, exist_ok=True)

for image_name in image_names:
    # get the path to the image
    image_path = os.path.join(images_path, image_name)
    
    # load the image and convert it to grayscale
    image = cv2.imread(image_path)
    print(image_name)

    # load the input image, convert it from BGR to Grayscale
    # and use Tesseract to localize each area of text in the input image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # run OCR on the image
    results = pytesseract.image_to_data(gray, output_type=Output.DICT)
    text = pytesseract.image_to_string(Image.open(image_path))
    print(text)

    # loop over each of the individual text localizations
    for i in range(len(results["text"])):
        # extract the bounding box coordinates of the text region from
        # the current result
        x = results["left"][i]
        y = results["top"][i]
        w = results["width"][i]
        h = results["height"][i]
        # extract the OCR text itself along with the confidence of the
        # text localization
        text = results["text"][i]
        conf = int(results["conf"][i])

        # filter out weak confidence text localizations
        if conf > min_confidence:
            # strip out non-ASCII text so we can draw the text on the image
            # using OpenCV, then draw a bounding box around the text along
            # with the text itself
            text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
    # save the output image
    output_path = os.path.join(output_dir, image_name)
    print(f"[INFO] saved output to {output_path}")
    print("------------------------------------")
    cv2.imwrite(output_path, image)


