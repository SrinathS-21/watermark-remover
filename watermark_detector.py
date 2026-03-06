import os
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

def detect_watermarks():
    # Load the model from Hugging Face
    try:
        print("Downloading model: mnemic/watermarks_yolov8...")
        model_path = hf_hub_download(repo_id="mnemic/watermarks_yolov8", filename="watermarks_s_yolov8_v1.pt")
        print(f"Model downloaded to: {model_path}")
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error downloading/loading model: {e}")
        print("Scanning repository for any .pt file...")
        try:
            from huggingface_hub import list_repo_files
            files = list_repo_files(repo_id="mnemic/watermarks_yolov8")
            pt_files = [f for f in files if f.endswith('.pt')]
            if pt_files:
                print(f"Found .pt files: {pt_files}")
                model_path = hf_hub_download(repo_id="mnemic/watermarks_yolov8", filename=pt_files[0])
                model = YOLO(model_path)
            else:
                print("No .pt files found in the repository.")
                return
        except Exception as e2:
            print(f"Failed to find model file: {e2}")
            return

    # Path to the dataset
    dataset_path = r"D:\pepagora\Watermark Remover\product sample images"
    output_path = r"D:\pepagora\Watermark Remover\detected_watermarks"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset path not found: {dataset_path}")
        return

    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created output directory: {output_path}")

    # Iterate through images
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    images = [f for f in os.listdir(dataset_path) if f.lower().endswith(image_extensions)]

    if not images:
        print("No images found in the dataset folder.")
        return

    print(f"Found {len(images)} images. Starting detection...\n")

    for image_name in images:
        image_path = os.path.join(dataset_path, image_name)
        
        # Run inference and save annotated image
        # save=True saves to 'runs/detect/predict' by default, 
        # so we'll use project/name to control the output location
        results = model.predict(image_path, project=output_path, name="annotated", exist_ok=True, save=True, verbose=False)

        for result in results:
            boxes = result.boxes
            if len(boxes) == 0:
                continue
            
            print(f"Detected {len(boxes)} watermark(s) in: {image_name}")
            for box in boxes:
                # Get coordinates (x1, y1, x2, y2)
                coords = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = box.cls[0].item()
                label = model.names[int(cls)]
                
                print(f"  - Label: {label}, Confidence: {conf:.2f}, Coordinates: {coords}")
        print("-" * 30)

    print(f"\nDetection complete. Annotated images are stored in: {os.path.join(output_path, 'annotated')}")

if __name__ == "__main__":
    detect_watermarks()