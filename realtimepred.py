import numpy as np
from tensorflow.keras.models import load_model
import cv2

# Load the trained model
model = load_model("model.h5")

# Map class indices to gesture names
class_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'none']

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw rectangle for hand gesture capture area
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
    cropped_frame = frame[100:300, 100:300]

    # Preprocess the cropped frame for prediction
    img = cv2.resize(cropped_frame, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Make prediction
    prediction = model.predict(img)
    class_index = np.argmax(prediction, axis=1)[0]
    class_label = class_names[class_index]
    confidence = np.max(prediction)

    # Display the prediction on the frame
    text = f"{class_label} ({confidence:.2f})"
    text = text.encode('ascii', 'ignore').decode('ascii')  # Ensures ASCII-only encoding
    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow('ISL Recognition', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

