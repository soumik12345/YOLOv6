import cv2
import wandb


class WandbInferenceLogger:
    def __init__(self) -> None:
        self.label_dictionary = {}
        self.table = wandb.Table(
            columns=[
                "Image-File",
                "Predictions",
                "Number-of-Objects",
                "Prediction-Confidence",
            ]
        )

    def set_label_dictionary(self, label_dictionary):
        self.label_dictionary = label_dictionary

    def in_infer(self, image, image_file, detection_results):
        bbox_data, confidences = [], []
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        for idx, (*xyxy, confidence, class_id) in enumerate(detection_results):
            confidences.append(float(confidence))
            xyxy = [int(coord) for coord in xyxy]
            bbox_data.append(
                {
                    "position": {
                        "minX": xyxy[0] / width,
                        "maxX": xyxy[2] / width,
                        "minY": xyxy[1] / height,
                        "maxY": xyxy[3] / height,
                    },
                    "class_id": int(class_id),
                    "box_caption": f"Key {idx}: {self.label_dictionary[int(class_id)]} {float(confidence)}",
                    "scores": {"confidence": float(confidence)},
                }
            )
        self.table.add_data(
            image_file,
            wandb.Image(
                image,
                boxes={
                    "predictions": {
                        "box_data": bbox_data,
                        "class_labels": self.label_dictionary,
                    }
                },
            ),
            len(detection_results),
            confidences,
        )

    def on_infer_end(self):
        wandb.log({"Inference": self.table})
