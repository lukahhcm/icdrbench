from typing import Optional

import regex as re

from ..base_op import OPERATORS, Mapper
from .pii_atomic_patterns import ID_CARD_CN_PATTERN


@OPERATORS.register_module("clean_id_card_mapper")
class CleanIdCardMapper(Mapper):
    """Clean ID-card numbers from text samples."""

    _batched_op = True

    def __init__(self, pattern: Optional[str] = None, repl: str = "", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pattern = pattern or ID_CARD_CN_PATTERN
        self.repl = repl

    def process_batched(self, samples):
        for idx, text in enumerate(samples[self.text_key]):
            if not re.search(self.pattern, text):
                continue
            samples[self.text_key][idx] = re.sub(self.pattern, self.repl, text)
        return samples
