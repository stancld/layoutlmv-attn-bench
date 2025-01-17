#!/usr/bin/env python
from __future__ import annotations

import argparse

import lightning.pytorch as pl
import torch
import torch.utils
import torch.utils.data
from transformers import LayoutLMv3Config, LayoutLMv3ForTokenClassification

from layoutlmv_attn_benchmark.data import BenchmarkData
from layoutlmv_attn_benchmark.training_module import TrainingModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attn_impl", type=str, choices=["eager", "sdpa"], default="eager")
    parser.add_argument("--model_name", type=str, default="microsoft/layoutlmv3-base")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=10000)

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    model_config = LayoutLMv3Config.from_pretrained(args.model_name)
    # Enforce turned off attention bias as it's not supported by SDPA
    model_config.has_relative_attention_bias = model_config.has_spatial_attention_bias = False
    # Explicitly define attention implementation
    model_config._attn_implementation = args.attn_impl
    model_config.num_labels = args.num_classes

    model = LayoutLMv3ForTokenClassification.from_pretrained(args.model_name, config=model_config)

    loader = torch.utils.data.DataLoader(
        BenchmarkData(
            args.num_samples, model_config.vocab_size, args.num_classes, args.max_length
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    training_module = TrainingModule(model)

    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(training_module, train_dataloaders=loader)


if __name__ == "__main__":
    main(parse_args())