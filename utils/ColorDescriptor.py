import numpy as np
import torch
import imutils


class ColorDescriptor:
    def __init__(self, bins):
        self.bins = bins

    def describe(self, image):
        # convert the image to the HSV color space
        features = []

        # grab the dimensions and compute the center of the image
        (h, w) = image.shape[-2:]
        (cX, cY) = (int(w * 0.5), int(h * 0.5))

        # top-left, top-right, bottom-right, bottom-left corner
        segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h), (0, cX, cY, h)]

        # elliptical mask in the center of the image
        (axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
        ellipMask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

        for (startX, endX, startY, endY) in segments:
            # construct a mask for each corner
            cornerMask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
            cornerMask = cv2.subtract(cornerMask, ellipMask)

            # extract a color histogram from the image
            hist = self.histogram(image, cornerMask)

            # then update the feature vector
            features.extend(hist)

        # extract a color histogram from the elliptical region
        hist = self.histogram(image, ellipMask)

        # then update the feature vector
        features.extend(hist)

        # return the feature vector
        return features

    def histogram(self, image, mask):
        # extract a 3D color histogram from the masked region
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        # return the histogram
        return hist


if __name__ == '__main__':
    from Dataset import show_tensor_images, ModalDataset

    path = '/Users/sanek_tarasov/Documents/École polytechnique/2A/P3/Modal'
    train_dataset = ModalDataset('train', path)
    descriptor = ColorDescriptor(1e3)

    # print(descriptor.describe(train_dataset[0]))
    torch.histogram(torch.tensor([1., 2, 1]), bins=4, range=(0., 3.), weight=torch.tensor([1., 2., 4.]))

