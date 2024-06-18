import easyocr
import cv2
import os
from config import *

# Initialize the EasyOCR reader with the English language model
reader = easyocr.Reader(['en'])

# Get the current working directory
cwd = os.getcwd()

# Get the path to the input folder and list all the image files
images_path = os.path.join(cwd, input_folder)
image_names = os.listdir(images_path)

# Create the output directory if it doesn't exist
output_dir = os.path.join(cwd, preprocessed_directory)
os.makedirs(output_dir, exist_ok=True)

for image_name in image_names:
    # get the path to the image
    image_path = os.path.join(images_path, image_name)
    
    # load the image and convert it to grayscale
    image = cv2.imread(image_path)
    print(image_name)

    img = cv2.imread(image_path)
    result = reader.readtext(img)

    for res in result:
        if res[2] < min_confidence:
            continue
        start_x, start_y = int(res[0][0][0]), int(res[0][0][1])
        end_x, end_y = int(res[0][2][0]), int(res[0][2][1])
        print(f" Text detected: {res[1]}, confidence: {res[2]:.3f}")
        cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        cv2.putText(img, res[1], (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
   
    output_path = os.path.join(output_dir, image_name)
    print(f"[INFO] saved output to {output_path}")
    cv2.imwrite(output_path, img)
    print("------------------------------------")
    print("")
    
