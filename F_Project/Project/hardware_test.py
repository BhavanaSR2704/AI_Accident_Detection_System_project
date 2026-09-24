from ultralytics import YOLO

print("Loading YOLO model...")

model = YOLO("yolo11n.pt")

print("Running accident video test...")

results = model.predict(
    source=r"C:\Users\shrey\OneDrive\Desktop\Project",
    show=True,
    save=True,
    verbose=True
)

print("Test completed!")