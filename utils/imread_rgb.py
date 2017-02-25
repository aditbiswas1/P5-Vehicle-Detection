import cv2

def imread_rgb(imagefile):
    image = cv2.imread(imagefile)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)