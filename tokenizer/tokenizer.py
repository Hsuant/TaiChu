"""Byte-Level BPE Tokenizer training service."""

from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from tokenizers import AddedToken, Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from tokenizer.config import ConfigError, TokenizerTrainingConfig, TrainerConfig, parse_config
from tokenizer.data_reader import TextRecordIterator
from tokenizer.decoder import DecoderFactory
from tokenizer.normalizer import NormalizerFactory
from tokenizer.pre_tokenizer import PreTokenizerFactory
from tokenizer.resource_monitor import ResourceMonitor, StageMetrics
from tokenizer.template import CHAT_TEMPLATE, validate_template_tokens
from tokenizer.validator import validate_tokenizer, validate_tokenizer_fast

logger = logging.getLogger(__name__)


class TrainerFactory:
    """Create the Byte-Level BPE trainer while preserving Token ID order."""

    @staticmethod
    def create(config: TrainerConfig) -> BpeTrainer:
        special_tokens = [
            AddedToken(token, normalized=False, lstrip=False, rstrip=False)
            for token in config.special_tokens.ordered_tokens
        ]
        return BpeTrainer(
            vocab_size=config.vocab_size,
            min_frequency=config.min_frequency,
            show_progress=config.show_progress,
            special_tokens=special_tokens,
            initial_alphabet=ByteLevel.alphabet(),
        )


class TaiChuTokenizerTrainer:
    """Own the Tokenizer lifecycle from streaming input to reload validation."""

    def __init__(
        self,
        config: TokenizerTrainingConfig | Mapping[str, Any],
        monitor: Optional[ResourceMonitor] = None,
    ):
        if isinstance(config, TokenizerTrainingConfig):
            self.config = config
        else:
            self.config = parse_config(dict(config))
        self.monitor = monitor or ResourceMonitor(
            self.config.runtime.monitor_interval_seconds
        )
        self.tokenizer: Optional[Tokenizer] = None
        self.validated_report: Dict[str, object] = {}

    def run(self) -> Dict[str, object]:
        self.monitor.start()
        run_started = time.perf_counter()
        try:
            with self.monitor.stage("initialize") as stage:
                tokenizer, trainer, chat_template = self._initialize()
                stage.chars = len(chat_template)

            corpus = TextRecordIterator(self.config.data)
            with self.monitor.stage("train_streaming_bpe") as stage:
                self._train(tokenizer, trainer, corpus, stage)
            tokenizer.decoder = DecoderFactory.create({"type": "bytelevel"})
            actual_vocab_size = tokenizer.get_vocab_size()

            with self.monitor.stage("save"):
                self._save(tokenizer, chat_template)

            del trainer, tokenizer
            self.tokenizer = None
            gc.collect()
            validation_mode = (
                "reload_validate_full"
                if self.config.runtime.full_reload_validation
                else "reload_validate_fast"
            )
            with self.monitor.stage(validation_mode):
                validator = (
                    validate_tokenizer
                    if self.config.runtime.full_reload_validation
                    else validate_tokenizer_fast
                )
                self.validated_report = validator(
                    self.config.save.directory,
                    self.config.trainer.special_tokens.ordered_tokens,
                    self.config.trainer.vocab_size,
                )
            report = {
                "tokenizer": str(self.config.save.directory),
                "configured_vocab_size": self.config.trainer.vocab_size,
                "actual_vocab_size": actual_vocab_size,
                "reader_stats": corpus.stats.to_dict(),
                "validation": self.validated_report,
                "elapsed_seconds": round(time.perf_counter() - run_started, 3),
                "resources": self.monitor.report(),
            }
            gc.collect()
            return report
        finally:
            self.monitor.stop()

    def _initialize(self):
        special = self.config.trainer.special_tokens
        tokenizer = Tokenizer(BPE(unk_token=special.unk_token))
        tokenizer.normalizer = NormalizerFactory.create(
            {
                "type": self.config.normalizer.type,
                "steps": list(self.config.normalizer.steps),
            }
        )
        tokenizer.pre_tokenizer = PreTokenizerFactory.create(
            {
                "type": "bytelevel",
                "add_prefix_space": self.config.pre_tokenizer.add_prefix_space,
                "use_regex": self.config.pre_tokenizer.use_regex,
                "digit_split": self.config.pre_tokenizer.digit_split,
            }
        )
        trainer = TrainerFactory.create(self.config.trainer)
        chat_template = CHAT_TEMPLATE
        validate_template_tokens(
            chat_template,
            self.config.trainer.special_tokens.ordered_tokens,
        )
        self.tokenizer = tokenizer
        logger.info(
            "BPE 初始化: vocab_size=%d min_frequency=%d special_tokens=%d",
            self.config.trainer.vocab_size,
            self.config.trainer.min_frequency,
            len(special.ordered_tokens),
        )
        return tokenizer, trainer, chat_template

    def _train(
        self,
        tokenizer: Tokenizer,
        trainer: BpeTrainer,
        corpus: TextRecordIterator,
        stage: StageMetrics,
    ) -> None:
        progress_every = self.config.runtime.progress_every_records

        def on_progress(stats) -> None:
            stage.records = stats.records_emitted
            stage.chars = stats.chars_emitted
            if stats.records_emitted % progress_every == 0:
                logger.info(
                    "Tokenizer 训练读取进度: records=%d chars=%d",
                    stats.records_emitted,
                    stats.chars_emitted,
                )

        corpus.on_progress = on_progress
        logger.info("开始 Byte-Level BPE 训练")
        tokenizer.train_from_iterator(corpus, trainer=trainer)
        stage.records = corpus.stats.records_emitted
        stage.chars = corpus.stats.chars_emitted

        special = self.config.trainer.special_tokens
        actual_ids = {
            token: tokenizer.token_to_id(token)
            for token in special.ordered_tokens
        }
        expected_ids = {
            token: index for index, token in enumerate(special.ordered_tokens)
        }
        if actual_ids != expected_ids:
            raise ConfigError(f"训练后的特殊 Token ID 不符合协议: {actual_ids}")
        actual_vocab_size = tokenizer.get_vocab_size()
        if actual_vocab_size != self.config.trainer.vocab_size:
            raise ConfigError(
                f"实际词表大小 {actual_vocab_size} 与目标 "
                f"{self.config.trainer.vocab_size} 不一致；请检查语料多样性或"
                "min_frequency，禁止静默保存与模型权重不匹配的词表"
            )
        logger.info(
            "BPE 训练完成: records=%d actual_vocab=%d elapsed=%.3fs",
            corpus.stats.records_emitted,
            actual_vocab_size,
            stage.elapsed_seconds,
        )

    def _save(self, tokenizer: Tokenizer, chat_template: str) -> None:
        special = self.config.trainer.special_tokens
        directory = self.config.save.directory
        if not self.config.save.overwrite and (directory / "tokenizer.json").exists():
            raise ConfigError(
                f"输出目录已存在 Tokenizer 且 save.overwrite=false: {directory}"
            )
        directory.mkdir(parents=True, exist_ok=True)

        tokenizer.save(str(directory / "tokenizer.json"))
        (directory / "chat_template.jinja").write_text(chat_template, encoding="utf-8")
        self._write_tokenizer_config(directory, chat_template)
        self._write_special_tokens_map(directory)
        logger.info("Tokenizer 已保存: %s", directory)

    def _write_tokenizer_config(self, directory: Path, chat_template: str) -> None:
        special = self.config.trainer.special_tokens
        saved = {
            "add_bos_token": False,
            "add_eos_token": False,
            "add_prefix_space": self.config.pre_tokenizer.add_prefix_space,
            "backend": "tokenizers",
            "bos_token": special.bos_token,
            "clean_up_tokenization_spaces": False,
            "eos_token": special.eos_token,
            "unk_token": special.unk_token,
            "pad_token": special.pad_token,
            "additional_special_tokens": list(special.additional_special_tokens),
            "added_tokens_decoder": {
                str(token_id): {
                    "content": token,
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True,
                }
                for token_id, token in enumerate(special.ordered_tokens)
            },
            "model_max_length": self.config.save.model_max_length,
            "tokenizer_class": "PreTrainedTokenizerFast",
            "chat_template": chat_template,
        }
        if self.config.model_meta:
            saved["model_info"] = self.config.model_meta
        with (directory / "tokenizer_config.json").open("w", encoding="utf-8") as handle:
            json.dump(saved, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    def _write_special_tokens_map(self, directory: Path) -> None:
        special = self.config.trainer.special_tokens
        special_map = {
            "bos_token": special.bos_token,
            "eos_token": special.eos_token,
            "unk_token": special.unk_token,
            "pad_token": special.pad_token,
            "additional_special_tokens": list(special.additional_special_tokens),
        }
        with (directory / "special_tokens_map.json").open("w", encoding="utf-8") as handle:
            json.dump(special_map, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
