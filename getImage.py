import cv2 as cv

# Get image
carImage = cv.imread('./assets/car.png')

# Transform image to grayscale
transformCarImageToGrayscale = cv.cvtColor(carImage, cv.COLOR_BGR2GRAY)

# Change shades of white and black
ret, transformCarImageToBinary = cv.threshold(transformCarImageToGrayscale, 90, 255, cv.THRESH_BINARY)

# Apply blur to image 
applyBlurToCarImage = cv.GaussianBlur(transformCarImageToBinary, (3, 3), cv.BORDER_REPLICATE)

# Find contours of image
contours, hier = cv.findContours(applyBlurToCarImage, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

# Draw contours
# cv.drawContours(carImage, contours, -1, (0, 255, 0), 2)

# Find square contours
for contour in contours: 
    perimeter = cv.arcLength(contour, True)

    if perimeter > 120:
        approx = cv.approxPolyDP(contour, 0.03 * perimeter, True)

        if len(approx) == 4: 
            (x, y, height, width) = cv.boundingRect(contour)
            cv.rectangle(carImage, (x, y), (x + height, y + width), (0, 255, 0), 2)

            snippingLicensePlate = carImage[y: y + width, x: x + height]

            cv.imwrite('./assets/snippingLicensePlate.png', snippingLicensePlate)
        
# Show image
cv.imshow('Image', carImage)

# Manter janela da imagem aberta, até que uma tecla seja pressionada
cv.waitKey(0)


