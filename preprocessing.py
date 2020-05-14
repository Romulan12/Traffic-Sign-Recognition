import cv2

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img = cv2.equalizeHist(img)
    return img
def preprocess(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img



