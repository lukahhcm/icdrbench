from typing import Optional

import regex as re

from ..base_op import OPERATORS, Mapper
from .pii_atomic_patterns import PHONE_CN_PATTERN, PHONE_INTL_PATTERN


@OPERATORS.register_module("clean_phone_mapper")
class CleanPhoneMapper(Mapper):
    """Clean phone numbers from text samples."""

    _batched_op = True

    def __init__(
        self,
        cn_pattern: Optional[str] = None,
        intl_pattern: Optional[str] = None,
        repl: str = "",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cn_pattern = cn_pattern or PHONE_CN_PATTERN
        self.intl_pattern = intl_pattern or PHONE_INTL_PATTERN
        self.repl = repl

    def process_batched(self, samples):
        for idx, text in enumerate(samples[self.text_key]):
            updated = re.sub(self.cn_pattern, self.repl, text)
            updated = re.sub(self.intl_pattern, self.repl, updated)
            if updated != text:
                samples[self.text_key][idx] = updated
        return samples
