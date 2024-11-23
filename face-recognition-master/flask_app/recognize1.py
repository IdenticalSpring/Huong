import cv2
import numpy as np
import torch
import yaml
from torchvision import transforms
from face_alignment.alignment import norm_crop
from face_detection.scrfd.detector import SCRFD
from face_tracking.tracker.byte_tracker import BYTETracker
from face_tracking.tracker.visualize import plot_tracking
from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import compare_encodings, read_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FaceRecognizer:
    def __init__(self, config_path):
        self.detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")
        self.recognizer = iresnet_inference(
            model_name="r100",
            path="face_recognition/arcface/weights/arcface_r100.pth",
            device=device,
        )
        self.config = self._load_config(config_path)
        self.tracker = BYTETracker(args=self.config, frame_rate=30)
        self.images_names, self.images_embs = read_features(
            feature_path="./datasets/face_features/feature"
        )
        self.data_mapping = {
            "raw_image": [],
            "tracking_ids": [],
            "detection_bboxes": [],
            "detection_landmarks": [],
            "tracking_bboxes": [],
        }
        self.id_face_mapping = {}

    def _load_config(self, file_name):
        with open(file_name, "r") as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    @torch.no_grad()
    def _get_feature(self, face_image):
        face_preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((112, 112)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_image = face_preprocess(face_image).unsqueeze(0).to(device)
        emb_img_face = self.recognizer(face_image).cpu().numpy()
        return emb_img_face / np.linalg.norm(emb_img_face)

    def recognize_face(self, face_image):
        query_emb = self._get_feature(face_image)
        score, id_min = compare_encodings(query_emb, self.images_embs)
        name = self.images_names[id_min]
        return score[0], name

    def _process_tracking(self, frame, frame_id, fps):
        outputs, img_info, bboxes, landmarks = self.detector.detect_tracking(image=frame)
        tracking_bboxes = []
        tracking_tlwhs = []
        tracking_ids = []

        if outputs is not None:
            online_targets = self.tracker.update(
                outputs, [img_info["height"], img_info["width"]], (128, 128)
            )
            for target in online_targets:
                tlwh = target.tlwh
                tid = target.track_id
                if tlwh[2] * tlwh[3] > self.config["min_box_area"]:
                    x1, y1, w, h = tlwh
                    tracking_bboxes.append([x1, y1, x1 + w, y1 + h])
                    tracking_tlwhs.append(tlwh)
                    tracking_ids.append(tid)

        tracking_image = plot_tracking(
            img_info["raw_img"],
            tracking_tlwhs,
            tracking_ids,
            names=self.id_face_mapping,
            frame_id=frame_id,
            fps=fps,
        )

        self.data_mapping["raw_image"] = img_info["raw_img"]
        self.data_mapping["detection_bboxes"] = bboxes
        self.data_mapping["detection_landmarks"] = landmarks
        self.data_mapping["tracking_ids"] = tracking_ids
        self.data_mapping["tracking_bboxes"] = tracking_bboxes

        return tracking_image

    def process_frame(self, frame, frame_id, fps):
        tracking_image = self._process_tracking(frame, frame_id, fps)
        detection_landmarks = self.data_mapping["detection_landmarks"]
        detection_bboxes = self.data_mapping["detection_bboxes"]
        tracking_ids = self.data_mapping["tracking_ids"]
        tracking_bboxes = self.data_mapping["tracking_bboxes"]

        for i in range(len(tracking_bboxes)):
            for j in range(len(detection_bboxes)):
                face_alignment = norm_crop(
                    img=self.data_mapping["raw_image"],
                    landmark=detection_landmarks[j],
                )
                score, name = self.recognize_face(face_image=face_alignment)
                caption = "UN_KNOWN" if score < 0.25 else f"{name}:{score:.2f}"
                self.id_face_mapping[tracking_ids[i]] = caption

        return tracking_image
