from __future__ import annotations

import torch


class BenchmarkData(torch.utils.data.Dataset):
    def __init__(
        self, num_samples: int, vocab_size: int, num_classes: int, max_length: int = 512
    ) -> None:
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_classes = num_classes

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.randint(0, self.vocab_size, (self.max_length,)),
            "bbox": torch.ones(self.max_length, 4, dtype=torch.int64),
            "attention_mask": torch.ones(self.max_length),
            "labels": torch.randint(0, self.num_classes, (self.max_length,)),
        }

    def __len__(self) -> int:
        return self.num_samples
