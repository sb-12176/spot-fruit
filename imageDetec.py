from ultralytics import YOLO
import cv2

# 1. Load your trained or pre-trained YOLO model
model = YOLO("C:\\fruitDataset\\runs\\detect\\train-2\\weights\\best.pt")  # Replace with your custom "best.pt" path if needed

# 2. Run inference on an image
results = model("C:\\fruitDataset\\blueberryTEST.png")

# 3. Get the annotated image array (returns a BGR numpy array)
annotated_image = results[0].plot()

# 4. Display the annotated image using OpenCV
cv2.imshow("YOLO Detections", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 5. Optional: Save the annotated image to a file
cv2.imwrite("annotated_output.jpg", annotated_image)
