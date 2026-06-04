import os
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    root = Path(__file__).resolve().parent
    data_yaml = root / "archive" / "data.yaml"

    model_name = os.getenv("YOLO_MODEL", "yolo11n.pt")
    epochs = int(os.getenv("EPOCHS", "50"))
    imgsz = int(os.getenv("IMGSZ", "640"))
    batch = int(os.getenv("BATCH", "16"))
    workers = int(os.getenv("WORKERS", "4"))
    patience = int(os.getenv("PATIENCE", "20"))
    project = os.getenv("PROJECT", "archive_runs")
    name = os.getenv("NAME", "train")

    model = YOLO(model_name)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        patience=patience,
        project=project,
        name=name,
        # You can tweak augmentations if you want:
        # degrees=0.0,
        # translate=0.1,
        # scale=0.5,
    )

    # Print a compact summary path
    try:
        best = results.best
        print(f"Best weights: {best}")
    except Exception:
        pass


if __name__ == "__main__":
    main()

