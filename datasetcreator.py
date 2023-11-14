import cv2

# Load the Haar Cascade classifier for face detection
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)
sampleNum = 0

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        sampleNum = sampleNum + 1
        cv2.imwrite("dataset/Ari_big_sig/x" + str(sampleNum) + ".jpg", img[y:y + h, x:x + w])
        cv2.imshow('frame', img)
    
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    elif sampleNum > 100:
        break

cam.release()
cv2.destroyAllWindows()
