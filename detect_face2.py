import pathlib
import cv2

def draw_boundry(img, classifier, scaleFactor, minNeighbors, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, text, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords, img

def detect(img, faceCascade):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0)}
    coords, img = draw_boundry(img, faceCascade, 1.1, 10, color['blue'], "Face")
    return img

# Utilise la webcam
camera = cv2.VideoCapture(0)

# Charge le classifieur
cascade_path = pathlib.Path(cv2.__file__).parent / "data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(str(cascade_path))

while True:
    ret, img = camera.read()
    if not ret:
        print("Erreur: image non capturée")
        break
    img = detect(img, faceCascade)
    cv2.imshow("Face detection", img)
    if cv2.waitKey(1) == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
