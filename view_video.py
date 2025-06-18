import cv2

# Load video
cap = cv2.VideoCapture("runs/segment/predict2/test_video.avi")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("AVI Video", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
