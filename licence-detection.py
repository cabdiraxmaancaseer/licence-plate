import cv2
import imutils
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract'
image = cv2.imread('nambarka.jpg')
image = imutils.resize(image, width=300 )
cv2.imshow("original image", image)


gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("greyed image", gray_image)


gray_image = cv2.bilateralFilter(gray_image, 11, 17, 17) 
cv2.imshow("smoothened image", gray_image)


edged = cv2.Canny(gray_image, 30, 200) 
cv2.imshow("edged image", edged)


cnts,new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
image1=image.copy()

cv2.drawContours(image=image1, contours=cnts, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
cv2.imshow("contours",image1)


cnts = sorted(cnts, key = cv2.contourArea, reverse = True) [:30]
screenCnt = None
image2 = image.copy()
cv2.drawContours(image2,cnts,-1,(125,255,0),6)
cv2.imshow("Top 30 contours",image2)


i=7
for c in cnts:
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.01 * perimeter, True)
    if len(approx) == 4:
    	screenCnt = approx
    	x,y,w,h = cv2.boundingRect(c)
    	new_img=image[y:y+h,x:x+w]
    	cv2.imwrite('./'+str(i)+'.png',new_img)
    	i+=1
    	break



cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
cv2.imshow("image with detected license plate", image)


Cropped_loc = './7.png'
cv2.imshow("cropped", cv2.imread(Cropped_loc))
plate = pytesseract.image_to_string(Cropped_loc, lang='eng')
print("Number plate is:", plate)
cv2.waitKey(0)
cv2.destroyAllWindows()