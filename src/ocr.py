#!/usr/bin/env python

import argparse
import cv2 as cv
import os
import sys
import glob
import math
import pytesseract
import imutils
import numpy as np
import re
import coloredlogs, logging

from rect import Rect

logger = logging.getLogger(__name__)
date_format = "%-H:%M:%S"
log_format = "%(asctime)s [%(levelname)s] %(message)s"
coloredlogs.install(datefmt=date_format, fmt=log_format, level=logging.INFO, logger=logger)

handler = logging.FileHandler(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../", "results.log"))
formatter = logging.Formatter(log_format, date_format)
handler.setFormatter(formatter)
# handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

class ParsedText(object):
    def __init__(self, text, rect):
        super(ParsedText, self).__init__()
        self.text = text
        self.rect = rect

    def __str__(self):
        return '"{}" @ {}'.format(self.text, self.rect)

    def __repr__(self):
        return '<ParsedText {}>'.format(str(self))

class TextLayout(object):
    # 0  - Orientation and script detection (OSD) only.
    # 1  - Automatic page segmentation with OSD.
    # 2  - Automatic page segmentation, but no OSD, or OCR.
    # 3  - Fully automatic page segmentation, but no OSD. (Default)
    # 4  - Assume a single column of text of variable sizes.
    # 5  - Assume a single uniform block of vertically aligned text.
    # 6  - Assume a single uniform block of text.
    # 7  - Treat the image as a single text line.
    # 8  - Treat the image as a single word.
    # 9  - Treat the image as a single word in a circle.
    # 10 - Treat the image as a single character.
    # 11 - Sparse text. Find as much text as possible in no particular order.
    # 12 - Sparse text with OSD.
    # 13 - Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific.

    COLUMN = 4
    UNIFORM_BLOCK_VALIGN = 5
    UNIFORM_BLOCK = 6
    LINE = 7
    WORD = 8
    SPARSE = 11
    RAW_LINE = 13

class TextRecognizer(object):
    WORKING_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")
    TEXT_DETECTION_MODEL_PATH = os.path.join(WORKING_DIR, "data/text_detection.pb")
    TESSERACT_DATA_DIR = os.path.join(WORKING_DIR, "data")
    IMAGES_DIR = os.path.join(WORKING_DIR, "assets/images")
    OUTPUT_DIR = os.path.join(WORKING_DIR, "assets/output")
    MATCHES_DIR = os.path.join(WORKING_DIR, "assets/matches")
    VIDEOS_DIR = os.path.join(WORKING_DIR, "assets/videos")
    TESSERACT_OEM = 1
    DEFAULT_LAYOUT = TextLayout.LINE

    def __init__(self, min_conf=0.5, max_angle=10, min_nms=0.4, padding=0.1):
        super(TextRecognizer, self).__init__()
        self.network = cv.dnn.readNet(self.TEXT_DETECTION_MODEL_PATH)
        self.min_conf = min_conf
        self.max_angle = max_angle
        self.min_nms = min_nms
        self.padding = max(0.0, min(padding, 1.0))

    def find_text(self, search_img, layout=DEFAULT_LAYOUT):
        located_rects = self._locate_text(search_img)
        results = self._parse_text(search_img, located_rects, layout)
        return results

    def _locate_text(self, search_img):
        img_rect = Rect(0, 0, search_img.shape[1], search_img.shape[0])

        # Input dimensions must be multiple of 32, image will be resized to these dimensions before running detection
        input_w = ((img_rect.w -1) | 31) + 1
        input_h = ((img_rect.h -1) | 31) + 1

        # Scale output rect dimensions by these amounts to transform coordinates back to source image's resolution
        rect_scale = (img_rect.w / float(input_w), img_rect.h / float(input_h))

        self.network.setInput(cv.dnn.blobFromImage(search_img, 1.0, (input_w, input_h), (123.68, 116.78, 103.94), True, False))
        output = self.network.forward([ "feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3" ])

        rects, confidences = self._compute_rects(output[0], output[1])
        indices = cv.dnn.NMSBoxes([list(rect) for rect in rects], confidences, self.min_conf, self.min_nms)

        results = []
        for [idx] in indices:
            result_rect = rects[idx].copy()
            result_rect.scale(rect_scale)
            
            pad_x = self.padding * 0.5 * result_rect.w
            pad_y = self.padding * 0.5 * result_rect.h

            result_rect.pad(pad_x, pad_y)
            result_rect.crop(Rect(0, 0, img_rect.w, img_rect.h))
            result_rect.map_to_pixels()
            results.append(result_rect)

        if not results:
            results = [img_rect]

        return results

    def _parse_text(self, search_img, search_rects, text_layout, filename):
        options = r'--tessdata-dir "{}" --oem {} --psm {}'.format(self.TESSERACT_DATA_DIR, self.TESSERACT_OEM, text_layout)
        results = []
        
        for rect in search_rects:
            grayscale = cv.imread(filename, cv.IMREAD_GRAYSCALE)
            median = cv.medianBlur(grayscale, 1)
            blurred = cv.GaussianBlur(median, (5,5), 0)
            cropped = rect.crop_image(blurred)
            resized = imutils.resize(cropped, width=400)

            (_, image) = cv.threshold(resized, 128, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

            result_text = pytesseract.image_to_string(image, lang="eng", config=options)

            matched_filename = os.path.basename(filename).replace(".", "-" + result_text + ".")
            save_dir = os.path.join(self.MATCHES_DIR, matched_filename)
            cv.imwrite(save_dir, image)

            if result_text is not None:
                results.append(ParsedText(result_text, rect))

        return results

    def _compute_rects(self, scores, geometry):
        rects = []
        confidences = []

        nrows, ncols = scores.shape[2:4]
        for row in range(0, nrows):
            confs = scores[0, 0, row]            
            x0_data = geometry[0, 0, row]
            x1_data = geometry[0, 1, row]
            x2_data = geometry[0, 2, row]
            x3_data = geometry[0, 3, row]
            angles = geometry[0, 4, row]

            for col in range(0, ncols):
                # Filter rects with low confidence or too great of an angle
                if confs[col] < self.min_conf or abs(math.degrees(angles[col])) > self.max_angle:
                    continue

                # Rotated bounding box width and height
                width = x1_data[col] + x3_data[col]
                height = x0_data[col] + x2_data[col]

                cos_a = math.cos(angles[col])
                sin_a = math.sin(angles[col])

                # Calculate rotate bounding rectangle
                offset = ([col * 4 + cos_a * x1_data[col] + sin_a * x2_data[col], row * 4 - sin_a * x1_data[col] + cos_a * x2_data[col]])
                pt1 = (-sin_a * height + offset[0], -cos_a * height + offset[1])
                pt3 = (-cos_a * width + offset[0], sin_a * width + offset[1])
                center = (0.5 * (pt1[0] + pt3[0]), 0.5 * (pt1[1] + pt3[1]))
                rotated_rect = (center, (width, height), -1 * math.degrees(angles[col]))

                # Convert rotated rect into axis-aligned containing bounding box
                corners = cv.boxPoints(rotated_rect)
                aabb = cv.boundingRect(corners)

                rects.append(Rect(aabb))
                confidences.append(float(confs[col]))

        return rects, confidences

def main():
    # python src/ocr.py images/player-names.png
    parser = argparse.ArgumentParser()
    parser.add_argument("input",                type=str,                               help="Path to input image")
    parser.add_argument("-c", "--min-conf",     type=float, default=0.5,                help="Filter out results with confidence less than this value")
    parser.add_argument("-a", "--max-angle",    type=float, default=10,                 help="Filter out results with an angle greater than this value")
    parser.add_argument("-n", "--min-nms",      type=float, default=0.4,                help="Filter out results with a non-maximum supression score less than this value")
    parser.add_argument("-p", "--padding",      type=float, default=0.1,                help="Amount of padding to apply to dtected text rectangles, as a percentage of the rectangle dimensions")
    parser.add_argument("-l", "--psm",          default=TextRecognizer.DEFAULT_LAYOUT,  help="Use specificed PSM Layout mode when running tesseract")
    parser.add_argument("-d", "--display",      action="store_true",                    help="Display the result in a GUI window using display X")
    parser.add_argument("-o", "--output",       type=str, nargs="?", const=True,        help="Save the generated image to a file, defaults to <filename>.ocr.<text>")
    parser.add_argument("-t", "--test",         type=str, nargs="+",                    help="Test for the occurrence of a string in the input image. Exits with return code 0 if found, 1 otherwise")
    args = parser.parse_args()
    
    original_image = cv.imread(args.input)
    output_image = original_image.copy()

    recognizer = TextRecognizer(args.min_conf, args.max_angle, args.min_nms, args.padding)
    text_rects = recognizer._locate_text(original_image)

    # Red boxes around located text
    for found_rect in text_rects:
        logger.debug("Located text at {}".format(found_rect))
        cv.rectangle(output_image, found_rect.top_left, found_rect.bottom_right, (255, 0, 0), 2, cv.LINE_AA)

    if hasattr(args.psm, "isdigit") and (args.psm).isdigit() != True:
        psm = args.psm.upper().replace(" ", "_")
        if hasattr(TextLayout, psm):
            logger.debug("Using PSM value '" + args.psm + "'")
            args.psm = TextLayout.__dict__[psm]
        else:
            logger.error("PSM value '" + args.psm + "' is in not accepted by OpenCV")

    parsed_text = recognizer._parse_text(original_image, text_rects, args.psm, args.input)

    # Green boxes around parsed text w/ parsed text printed in purple
    for parsed in parsed_text:
        logger.debug("Parsed text '" + parsed.text + "' at {}".format(parsed.rect))
        cv.rectangle(output_image, parsed.rect.top_left, parsed.rect.bottom_right, (0, 255, 0), 1, cv.LINE_AA)

        cv.putText(output_image, parsed.text, parsed.rect.top_left, cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2, cv.LINE_AA)

    # if args.output:
        # output_path = args.output
        # if output_path is True:
        #     filename, ext = os.path.splitext(args.input)
        #     output_path = "{}.ocr{}".format(filename, ext if ext else "")

    save_dir = os.path.join(TextRecognizer.OUTPUT_DIR, os.path.basename(args.input))
    cv.imwrite(save_dir, output_image)
    logger.info("📁 Output image save to " + save_dir)

    if args.test:
        for value in args.test:
            found_text = False
            text_re = re.compile(value, re.IGNORECASE)

            for parsed in parsed_text:
                if text_re.match(parsed.text):
                    (x, y, _, _) = parsed.rect
                    logger.info("✔️  Found '" + value + "' at " + "(X: " + str(x) + ", Y: " + str(y) + ")")
                    found_text = True
                    break

            if not found_text:
                logger.info("❌ Absent '" + value + "'")

    if args.display:
        cv.imshow("", output_image)
        while True:
            if cv.waitKey(0) == ord('q'):
                break

if __name__ == "__main__":
    main()