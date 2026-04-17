from typing import Optional

import regex as re

from ..base_op import OPERATORS, Mapper
from .pii_atomic_patterns import CHANNEL_KV_PATTERN, FEISHU_OPEN_ID_PATTERN, PLATFORM_OPEN_ID_PATTERN


@OPERATORS.register_module("clean_channel_id_mapper")
class CleanChannelIdMapper(Mapper):
    """Clean agent-channel identifiers and platform open IDs from text samples."""

    _batched_op = True

    def __init__(
        self,
        channel_pattern: Optional[str] = None,
        feishu_open_id_pattern: Optional[str] = None,
        platform_open_id_pattern: Optional[str] = None,
        repl: str = "",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.channel_pattern = channel_pattern or CHANNEL_KV_PATTERN
        self.feishu_open_id_pattern = feishu_open_id_pattern or FEISHU_OPEN_ID_PATTERN
        self.platform_open_id_pattern = platform_open_id_pattern or PLATFORM_OPEN_ID_PATTERN
        self.repl = repl

    def process_batched(self, samples):
        for idx, text in enumerate(samples[self.text_key]):
            updated = re.sub(self.channel_pattern, r"\1" + self.repl, text, flags=re.IGNORECASE)
            updated = re.sub(self.feishu_open_id_pattern, self.repl, updated, flags=re.IGNORECASE)
            updated = re.sub(self.platform_open_id_pattern, self.repl, updated)
            if updated != text:
                samples[self.text_key][idx] = updated
        return samples
