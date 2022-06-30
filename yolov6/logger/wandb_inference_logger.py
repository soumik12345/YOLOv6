import cv2
import wandb


class WandbInferenceLogger:
    def __init__(self) -> None:
        self.label_dictionary = {}

    def set_label_dictionary(self, label_dictionary):
        self.label_dictionary = label_dictionary

    def in_infer(self, image, detection_results):
        bbox_data = []
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        for *xyxy, confidence, class_id in detection_results:
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
                    "box_caption": self.label_dictionary[int(class_id)],
                    "scores": {"confidence": float(confidence)},
                }
            )
        wandb.log(
            {
                "Inference Results": wandb.Image(
                    image,
                    boxes={
                        "predictions": {
                            "box_data": bbox_data,
                            "class_labels": self.label_dictionary,
                        }
                    },
                )
            }
        )
