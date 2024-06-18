input_folder = "input_images"
preprocessed_directory = "processed_output"
east_model = "frozen_east_text_detection.pb"

image_height = 320
image_width = 320

min_confidence = 0.1
nms_threshold = 0.3
text_threshold = 10


preprocessing = "thresh"
# preprocessing = "blur"