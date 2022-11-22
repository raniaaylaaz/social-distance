import numpy as np
import cv2
 
# Inisialisasi Package dan Metode yang digunakan untuk Human Detection
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

# Buka Video Stream
cap = cv2.VideoCapture("Live Record 1 okt 22.mp4")

# Simpen jadi output.avi
out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640,480))

while(True):
    # Ambil Frame-by-Frame Video Sumber
    ret, frame = cap.read()

    # Resize jadi Lebih Kecil biar gak makan banyak Resource 
    frame = cv2.resize(frame, (640, 480))
    # Ubah jadi Grayscale biar HoG mantap + Gak Makan Banyak Resources
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Pendeteksian Orang di Video
    # Buat Bounding Box Kalo Detect Orang
    boxes, weights = hog.detectMultiScale(frame, winStride=(8,8) )

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for (xA, yA, xB, yB) in boxes:
        # Warnain Jadi Ijo Framenya (RGB)
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                          (0, 255, 0), 2)
    
    # Simpen Hasil Gambaran Bounding Box
    out.write(frame.astype('uint8'))
    # Tunjukkin Outputnya
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
# and release the output
out.release()
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)
