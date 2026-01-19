from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo11n.pt')

    print("Rozpoczynam trening na 100 epok...")
    results = model.train(
        data='yolo/data.yaml', 
        epochs=100,
        imgsz=640,
        device=0,
        patience=20,
        batch=16,
    )

    print("Trening zakończony! Sprawdź folder runs/detect")
