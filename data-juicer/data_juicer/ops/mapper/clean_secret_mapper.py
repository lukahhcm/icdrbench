from typing import Optional

import regex as re

from ..base_op import OPERATORS, Mapper
from .pii_atomic_patterns import SECRET_KV_PATTERN


@OPERATORS.register_module("clean_secret_mapper")
class CleanSecretMapper(Mapper):
    """Clean secret-like key-value payloads from text samples."""

    _batched_op = True

    def __init__(self, pattern: Optional[str] = None, repl: str = "", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pattern = pattern or SECRET_KV_PATTERN
        self.repl = repl

    def process_batched(self, samples):
        for idx, text in enumerate(samples[self.text_key]):
            if not re.search(self.pattern, text, flags=re.IGNORECASE):
                continue
            samples[self.text_key][idx] = re.sub(
                self.pattern,
                r"\1" + self.repl,
                text,
                flags=re.IGNORECASE,
            )
        return samples
