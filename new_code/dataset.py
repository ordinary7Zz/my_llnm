from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertModel, BertTokenizer, AutoTokenizer


from new_code.utils import apply_normalization, build_sample_records


class LLNMDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        image_root: str | Path,
        patient_info_file: str | Path | None = None,
        radiomics_csv: str | Path | None = None,
        default_report: str = "",
        default_age: float = 50.0,
        default_sex: float = 1.0,
        shape_feature: str = "original_shape2D_Elongation",
        echo_feature: str = "original_firstorder_Mean",
        max_report_tokens: int = 300,
        norm_stats: dict[str, float] | None = None,
        image_size: int = 224,
        resize_size: int = 256,
    ):
        self.image_root = Path(image_root)
        self.max_report_tokens = max_report_tokens
        self.samples = build_sample_records(
            manifest_path=manifest_path,
            image_root=image_root,
            default_report=default_report,
            default_age=default_age,
            default_sex=default_sex,
            patient_info_file=patient_info_file,
            radiomics_csv=radiomics_csv,
            shape_feature=shape_feature,
            echo_feature=echo_feature,
        )
        if norm_stats is not None:
            self.samples = [apply_normalization(sample, norm_stats) for sample in self.samples]

        if not self.samples:
            raise ValueError(f"no usable samples found in manifest: {manifest_path}")

        self.transform = transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        )

        self.tokenizer = AutoTokenizer.from_pretrained("/mnt/wangbd8/workspace/ThyroidAgent/LLNM-Net/my_pretrained/bert-base-chinese")
        # self.tokenizer = BertTokenizer.from_pretrained("/mnt/wangbd8/workspace/ThyroidAgent/LLNM-Net/my_pretrained/bert-base-chinese")
        self.bert_model = BertModel.from_pretrained("/mnt/wangbd8/workspace/ThyroidAgent/LLNM-Net/my_pretrained/bert-base-chinese")
        self.bert_model.eval()

    def __len__(self) -> int:
        return len(self.samples)

    @torch.no_grad()
    def _encode_report(self, report: str) -> torch.Tensor:
        input_ids = self.tokenizer.encode(report, add_special_tokens=True, return_tensors="pt")
        outputs = self.bert_model(input_ids)
        last_hidden_state = outputs.last_hidden_state

        padding_length = self.max_report_tokens - last_hidden_state.shape[1]
        if padding_length > 0:
            padding_tensor = torch.zeros(
                (1, padding_length, last_hidden_state.shape[2]),
                dtype=last_hidden_state.dtype,
            )
            padded_outputs = torch.cat([last_hidden_state, padding_tensor], dim=1)
        else:
            padded_outputs = last_hidden_state[:, : self.max_report_tokens, :]

        return padded_outputs.squeeze(0).float().contiguous()

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image_path = self.image_root / sample["image"]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        rr = self._encode_report(sample["report"])
        demo = torch.tensor([sample["age"], sample["sex"]], dtype=torch.float32)
        img_fea = torch.tensor([sample["shape"], sample["echo"]], dtype=torch.float32)
        label = torch.tensor(sample["LNM_CN01"], dtype=torch.long)

        return {
            "image": image,
            "label": label,
            "rr": rr,
            "demo": demo,
            "img_fea": img_fea,
            "path": sample["image"],
        }
