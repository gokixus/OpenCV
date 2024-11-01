import cv2

haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img = cv2.imread("people.jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=3)

for(x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)

cv2.imshow("Face Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
