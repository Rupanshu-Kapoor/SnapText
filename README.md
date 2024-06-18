# SnapText

SnapText is a powerful and intuitive Optical Character Recognition application built using EasyOCR and OpenCV to seamlessly detect and extract text from image. Whether you need to process scanned documents, images with embedded text, or any other visual data, SnapText provides a reliable solution for all your text detection needs.


## Features

**Accurate Text Detection**: Utilizes EasyOCR's advanced text recognition capabilities to accurately detect and extract text from images.
**Multi-Language Support**: Currently supports English, with the potential to extend to other languages supported by EasyOCR.
**Confidence Filtering**: Filters detected text based on a minimum confidence level to ensure high-quality results.
**Visual Feedback**: Displays bounding boxes and detected text on the original image, providing clear visual feedback.
**Batch Processing**: Processes multiple images in a directory, saving the output images with detected text in a specified output directory.

## Installation

1. **Clone the repository**:
`git clone https://github.com/your-username/SnapText.git
cd SnapText`

2. **Install dependencies**:
`pip install -r requirements.txt`

3. **Place your images in the `input_images` directory**:

## Usage

1. **Configure the settings in the `config.py` file**:
- `input_folder`: The path to the directory containing the input images.
- `preprocessed_directory`: The path to the directory where the preprocessed images will be saved.
- `east_model`: The path to the frozen EAST text detection model.
- `min_confidence`: The minimum confidence level for text detection.

2. **Run the script**:
`python easyocr_recognisation.py`

## Future Enhancements

- Add support for multiple languages.
- Implement a graphical user interface (GUI) for easier use.
- Extend functionality to handle different image formats and resolutions

## Contributions
Contributions are welcome! If you have any ideas, suggestions, or issues, please open an issue or submit a pull request.