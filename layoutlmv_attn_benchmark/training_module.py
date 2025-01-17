from __future__ import annotations

from typing import TYPE_CHECKING

import lightning
import torch

if TYPE_CHECKING:
    from typing import Any

    from transformers import LayoutLMv3ForTokenClassification
    from transformers.modeling_outputs import TokenClassifierOutput


class TrainingModule(lightning.LightningModule):
    def __init__(self, model: LayoutLMv3ForTokenClassification, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model = model

    def forward(self, batch: dict[str, torch.Tensor]) -> TokenClassifierOutput:
        return self.model(**batch)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        model_output = self.forward(batch)
        return model_output.loss

    def configure_optimizers(self) -> torch.optim.optimizer.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=0.02)
