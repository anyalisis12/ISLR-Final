import cv2
import os

# Configuration

gesture_name = 'I'  # Change for each gesture you want to collect
save_dir = f'data/{gesture_name}'
os.makedirs(save_dir, exist_ok=True)

# Open the webcam
cap = cv2.VideoCapture(0)
image_count = 0
max_images = 200  # Collect 200 images per gesture

print(f"Collecting images for gesture: {gesture_name}. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw rectangle to guide the hand placement
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
    cropped_frame = frame[100:300, 100:300]

    # Display the frame
    cv2.imshow("Frame", frame)

    # Save the cropped frame
    if image_count < max_images:
        file_path = os.path.join(save_dir, f"{gesture_name}_{image_count}.jpg")
        cv2.imwrite(file_path, cropped_frame)
        image_count += 1
        print(f"Collected image {image_count}/{max_images}")
    else:
        print(f"Collected {max_images} images for gesture '{gesture_name}'.")
        break

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
