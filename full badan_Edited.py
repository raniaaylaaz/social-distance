import numpy as np
import cv2
 
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

# Buka Video Yang Mau di Proses
cap = cv2.VideoCapture('record ke 2.mp4')

# the output will be written to output.avi
out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (854,480))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # resizing for faster detection
    frame = cv2.resize(frame, (854, 480))
    # Bikin jadi Lebih Gelap Biar Gaada Pantulan Cahaya
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #contrast = 0.9
    #brightness = 1
    #frame[:,:,2] = np.clip(contrast * frame[:,:,2] + brightness, 0, 255)
    #frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    

    # detect people in the image
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(gray, winStride=(8,8), padding=(16,16), scale=1.1 )
    #Diatas Harus Ubah Parameter Parameter Ampe Paling Mantap Dah
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    #print(boxes)
    
    if(len(boxes) >= 2):
        for (xA, yA, xB, yB) in boxes:
        # display the detected boxes in the colour picture
            cv2.rectangle(frame, (xA, yA), (xB, yB),
                          (0, 0, 255), 2) # VALUE RGB (BLUE, GREEN, RED MAX 255)
            cv2.putText(frame, "Confidence : "+str(weights), (xA, yA-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        for (xA, yA, xB, yB) in boxes:
            cv2.rectangle(frame, (xA, yA), (xB, yB),
                          (0, 255, 0), 2)
            cv2.putText(frame, "Confidence : "+str(weights), (xA, yA-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(frame, "Jumlah Orang Terdeteksi : "+str(len(boxes)), (520, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Write the output video 
    out.write(frame.astype('uint8'))
    # Display the resulting frame
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
