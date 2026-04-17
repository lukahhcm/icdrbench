from typing import Optional

import regex as re

from ..base_op import OPERATORS, Mapper
from .pii_atomic_patterns import PATH_UNIX_PATTERN, PATH_WIN_PATTERN, PATH_WIN_UNC_PATTERN


@OPERATORS.register_module("clean_path_mapper")
class CleanPathMapper(Mapper):
    """Clean Unix/Windows/UNC file-system paths from text samples."""

    _batched_op = True

    def __init__(
        self,
        unix_pattern: Optional[str] = None,
        win_pattern: Optional[str] = None,
        unc_pattern: Optional[str] = None,
        repl: str = "",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.unix_pattern = unix_pattern or PATH_UNIX_PATTERN
        self.win_pattern = win_pattern or PATH_WIN_PATTERN
        self.unc_pattern = unc_pattern or PATH_WIN_UNC_PATTERN
        self.repl = repl

    def process_batched(self, samples):
        for idx, text in enumerate(samples[self.text_key]):
            updated = re.sub(self.unix_pattern, r"\1" + self.repl, text)
            updated = re.sub(self.win_pattern, r"\1" + self.repl, updated)
            updated = re.sub(self.unc_pattern, r"\1" + self.repl, updated)
            if updated != text:
                samples[self.text_key][idx] = updated
        return samples
