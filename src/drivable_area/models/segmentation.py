"""TwinLiteNet+ lane and SegFormer road masks; imported only for live inference."""

from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np


class SegmentationModels:
    def __init__(self, settings):
        import torch
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        from drivable_area._vendor.twinlitenet.model import TwinLiteNetPlus

        self.torch = torch
        self.settings = settings
        device = (
            "cuda"
            if settings.device == "auto" and torch.cuda.is_available()
            else "cpu"
            if settings.device == "auto"
            else settings.device
        )
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA를 사용할 수 없습니다. models.device='cpu' 또는 'auto'를 사용하세요"
            )
        self.device = torch.device(device)
        if device == "cpu":
            torch.set_num_threads(settings.cpu_threads)
        weight = Path(settings.twin_weight)
        if not weight.is_file():
            raise FileNotFoundError(
                f"TwinLiteNet+ 가중치가 없습니다: {weight}. weights/README.md를 확인하세요"
            )
        self.twin = (
            TwinLiteNetPlus(SimpleNamespace(config=settings.twin_config)).to(self.device).eval()
        )
        self.twin.load_state_dict(torch.load(weight, map_location=self.device, weights_only=True))
        kwargs = dict(cache_dir=settings.cache_dir, local_files_only=settings.local_files_only)
        self.processor = SegformerImageProcessor.from_pretrained(settings.model_id, **kwargs)
        self.segformer = (
            SegformerForSemanticSegmentation.from_pretrained(settings.model_id, **kwargs)
            .to(self.device)
            .eval()
        )
        ids = [
            int(i) for i, name in self.segformer.config.id2label.items() if name.lower() == "road"
        ]
        if len(ids) != 1:
            raise ValueError("SegFormer에 단일 road 클래스가 필요합니다")
        self.road_id = ids[0]

    def predict(self, image, output_size):
        from drivable_area._vendor.twinlitenet.preprocess import letterbox_for_img

        torch = self.torch
        h, w = image.shape[:2]
        with torch.inference_mode():
            padded, _, (pw, ph) = letterbox_for_img(image, self.settings.img_size, auto=True)
            tensor = (
                torch.from_numpy(np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)))
                .unsqueeze(0)
                .to(self.device, dtype=torch.float32)
                / 255.0
            )
            _, lane_logits = self.twin(tensor)
            left, right = round(pw - 0.1), round(pw + 0.1)
            top, bottom = round(ph - 0.1), round(ph + 0.1)
            lane_logits = lane_logits[
                :, :, top : lane_logits.shape[2] - bottom, left : lane_logits.shape[3] - right
            ]
            lane = (
                torch.nn.functional.interpolate(
                    lane_logits, size=(h, w), mode="bilinear", align_corners=False
                )
                .argmax(1)[0]
                .cpu()
                .numpy()
                == 1
            )
            ratio = self.settings.seg_size / max(h, w)
            inputs = self.processor(
                images=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                size={"height": max(32, round(h * ratio)), "width": max(32, round(w * ratio))},
                return_tensors="pt",
            ).to(self.device)
            logits = self.segformer(**inputs).logits
            best = torch.full((h, w), -float("inf"), device=self.device)
            labels = torch.zeros((h, w), dtype=torch.long, device=self.device)
            for index in range(logits.shape[1]):
                values = torch.nn.functional.interpolate(
                    logits[:, index : index + 1], size=(h, w), mode="bilinear", align_corners=False
                )[0, 0]
                wins = values > best
                labels[wins] = index
                best = torch.maximum(best, values)
            road = labels.cpu().numpy() == self.road_id
        return tuple(
            cv2.resize(mask.astype(np.uint8), output_size, interpolation=cv2.INTER_NEAREST) > 0
            for mask in (road, lane)
        )
