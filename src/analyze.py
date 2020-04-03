import pytesseract

def boxToString(image, boxes, padding=0):
    # initialize the list of results
    results = []
    # (height, width) = image.shape[:2]

    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # # in order to obtain a better OCR of the text we can potentially
        # # apply a bit of padding surrounding the bounding box -- here we
        # # are computing the deltas in both the x and y directions
        # dX = int((endX - startX) * padding)
        # dY = int((endY - startY) * padding)

        # # apply padding to each side of the bounding box, respectively
        # startX = max(0, startX - dX)
        # startY = max(0, startY - dY)
        # endX = min(height, endX + (dX * 2))
        # endY = min(width, endY + (dY * 2))

        # extract the actual padded ROI
        roi = image[startY:endY, startX:endX]
        # roi = image[endY:startY, startX:endX]
        # print(image.shape, startX, startY, endX, endY)

        # in order to apply Tesseract v4 to OCR text we must supply
        # (1) a language, (2) an OEM flag of 4, indicating that the we
        # wish to use the LSTM neural net model for OCR, and finally
        # (3) an OEM value, in this case, 7 which implies that we are
        # treating the ROI as a single line of text
        # config = ("-l eng --oem 3 --psm 7")
        text = pytesseract.image_to_string(roi)
        # print(text)

        # add the bounding box coordinates and OCR'd text to the list
        # of results
        results.append(((startX, startY, endX, endY), text))

    # sort the results bounding box coordinates from top to bottom
    results = sorted(results, key=lambda r:r[0][1])

    return results