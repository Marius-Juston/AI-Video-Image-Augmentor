INPUT_FILEPATH = "DAIN/input.mp4"  # @param{type:"string"}
OUTPUT_FILE_PATH = "DAIN/output.mp4"  # @param{type:"string"}
TARGET_FPS = 60  # @param{type:"number"}

FRAME_INPUT_DIR = '/content/DAIN/input_frames'  # @param{type:"string"}

FRAME_OUTPUT_DIR = '/content/DAIN/output_frames'  # @param{type:"string"}

SEAMLESS = False  # @param{type:"boolean"}

RESIZE_HOTFIX = True  # @param{type:"boolean"}

AUTO_REMOVE = True  # @param{type:"boolean"}

model = 'http://vllab1.ucmerced.edu/~wenbobao/DAIN/best.pth'

import os

filename = os.path.basename(INPUT_FILEPATH)

import cv2

cap = cv2.VideoCapture(f'/content/DAIN/{filename}')

fps = cap.get(cv2.CAP_PROP_FPS)

if (fps / TARGET_FPS > 0.5):
    print(
        "Define a higher fps, because there is not enough time for new frames. (Old FPS)/(New FPS) should be lower than 0.5. Interpolation will fail if you try.")

if (RESIZE_HOTFIX == True):
    images = []
    for filename in os.listdir(f'{FRAME_OUTPUT_DIR}'):
        img = cv2.imread(os.path.join(f'{FRAME_OUTPUT_DIR}', filename))
        part_filename = os.path.splitext(filename)
        if (part_filename[0].endswith('0') == False):
            dimension = (img.shape[1] + 2, img.shape[0] + 2)
            resized = cv2.resize(img, dimension, interpolation=cv2.INTER_LANCZOS4)
            crop = resized[1:(dimension[1] - 1), 1:(dimension[0] - 1)]
            cv2.imwrite(part_filename[0] + ".png", crop)
