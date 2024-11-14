import cv2
import os
from glob import glob

# Configuration
dataset_dir = 'data'  # Path to the dataset containing all alphabet folders
max_images = 200  # Set a limit to process up to 200 images per gesture

# Get all subdirectories (gestures) in the dataset directory
gesture_names = [gesture for gesture in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, gesture))]

# Loop through each gesture (alphabet)
for gesture_name in gesture_names:
    gesture_dir = os.path.join(dataset_dir, gesture_name)
    processed_dir = f'processed_data/{gesture_name}'
    os.makedirs(processed_dir, exist_ok=True)

    # Get a list of images in the dataset directory for the specified gesture
    image_paths = glob(os.path.join(gesture_dir, "*.jpg"))
    image_count = 0

    print(f"Processing images for gesture: {gesture_name}")

    # Process each image up to the specified limit
    for image_path in image_paths[:max_images]:  # Process up to max_images
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        # Resize the image (adjust the size as needed, here resizing to 200x200)
        processed_image = cv2.resize(image, (200, 200))

        # Save processed image to the new directory
        file_path = os.path.join(processed_dir, f"{gesture_name}_{image_count}.jpg")
        cv2.imwrite(file_path, processed_image)
        image_count += 1
        print(f"Processed image {image_count}/{max_images} for gesture '{gesture_name}'")

    print(f"Completed processing {image_count} images for gesture '{gesture_name}'.\n")

print("Processing complete for all gestures.")
