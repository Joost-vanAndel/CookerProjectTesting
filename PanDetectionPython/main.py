import numpy as np
import cv2 as cv

# load image
img = cv.imread('pics/Beans2.jpg')

# rescale to fit screen better
width = int(img.shape[1] * 0.3)
height = int(img.shape[0] * 0.3)
dim = (width, height)

resizedImg = cv.resize(img, dim, interpolation=cv.INTER_AREA)

# rotate image to be upright
rotatedImg = cv.rotate(resizedImg, cv.ROTATE_90_COUNTERCLOCKWISE)

# create grayscale image for houghcircles
grayImg = cv.cvtColor(rotatedImg, cv.COLOR_BGR2GRAY)

# blur image for beter results with houghcircles
blurredImg = cv.medianBlur(grayImg, 5)

# houghcircles to find circles in image
circles = cv.HoughCircles(blurredImg, cv.HOUGH_GRADIENT, 1, blurredImg.shape[0] / 16,
                          param1=120, param2=70, minRadius=260, maxRadius=350)

# draw circles
circleImg = rotatedImg
circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    # draw the outer circle
    cv.circle(circleImg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv.circle(circleImg, (i[0], i[1]), 2, (0, 0, 255), 3)

# create new image with circle content
maskedImg = cv.bitwise_and(rotatedImg, rotatedImg, mask=circles[0])

# show images
cv.imshow("Circles image", circleImg)
cv.imshow("Masked image", maskedImg)
cv.waitKey(0)


