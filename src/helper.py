
import cv2
import numpy as np
import time 

def resize_image(image, width, height):
    """
    Re-sizes image to given width & height
    :param image: image to resize
    :param width: new width
    :param height: new height
    :return: modified image, ratio of new & old height and width
    """
    h, w = image.shape[:2]

    ratio_w = w / width
    ratio_h = h / height

    image = cv2.resize(image, (width, height))

    return image, ratio_w, ratio_h


def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
    """
    Applies Non-Maximum Suppression to a list of bounding boxes

    :param boxes: Bounding boxes to apply NMS to
    :param probs: Probabilities corresponding to each bounding box
    :param overlapThresh: Overlap threshold for suppressing boxes
    :return: List of bounding boxes after NMS
    """
    # If there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # If the bounding boxes are integers, convert them to floats
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # Initialize the list of picked indexes
    pick = []

    # Extract the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute the area of the bounding boxes and grab the indexes to sort
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # If probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # Sort the indexes
    idxs = np.argsort(idxs)

    # Keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # Grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # Delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
                                              np.where(overlap > overlapThresh)[0])))

    # Return only the bounding boxes that were picked
    return boxes[pick].astype("int")


def forward_passer(net, image, layers, timing=True):
    """
    Returns results from a single pass on a Deep Neural Net for a given list of layers
    :param net: Deep Neural Net (usually a pre-loaded .pb file)
    :param image: image to do the pass on
    :param layers: layers to do the pass through
    :param timing: show detection time or not
    :return: results obtained from the forward pass
    """
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (w, h), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    start = time.time()
    net.setInput(blob)
    scores, geometry = net.forward(layers)
    end = time.time()

    if timing:
        print(f"[INFO] detection in {round(end - start, 2)} seconds")

    return scores, geometry


def box_extractor(scores, geometry, min_confidence):
    """
    Converts results from the forward pass to rectangles depicting text regions & their respective confidences
    :param scores: scores array from the model
    :param geometry: geometry array from the model
    :param min_confidence: minimum confidence required to pass the results forward
    :return: decoded rectangles & their respective confidences
    """
    num_rows, num_cols = scores.shape[2:4]
    rectangles = []
    confidences = []

    for y in range(num_rows):
        scores_data = scores[0, 0, y]
        x_data0 = geometry[0, 0, y]
        x_data1 = geometry[0, 1, y]
        x_data2 = geometry[0, 2, y]
        x_data3 = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]

        for x in range(num_cols):
            if scores_data[x] < min_confidence:
                continue

            offset_x, offset_y = x * 4.0, y * 4.0

            angle = angles_data[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            box_h = x_data0[x] + x_data2[x]
            box_w = x_data1[x] + x_data3[x]

            end_x = int(offset_x + (cos * x_data1[x]) + (sin * x_data2[x]))
            end_y = int(offset_y + (cos * x_data2[x]) - (sin * x_data1[x]))
            start_x = int(end_x - box_w)
            start_y = int(end_y - box_h)

            rectangles.append((start_x, start_y, end_x, end_y))
            confidences.append(scores_data[x])

    return rectangles, confidences