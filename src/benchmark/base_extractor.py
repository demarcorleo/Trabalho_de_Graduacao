from abc import ABC, abstractmethod
import re

from benchmark_config import (
    COLLAPSE_EXCESSIVE_BLANK_LINES,
    NORMALIZE_LINE_ENDINGS,
    REMOVE_TRAILING_SPACES,
)


class BaseExtractor(ABC):

    name = "base"

    def setup(self) -> None:
        """
        Inicialização executada uma única vez antes do benchmark
        da biblioteca.
        """
        pass

    def teardown(self) -> None:
        """
        Liberação opcional de recursos.
        """
        pass

    @abstractmethod
    def extract(self, file_path: str) -> str:
        raise NotImplementedError

    def run(self, file_path: str) -> str:

        text = self.extract(file_path)

        if text is None:
            text = ""

        if not isinstance(text, str):
            text = str(text)

        return self.normalize_text(text)

    @staticmethod
    def normalize_text(text: str) -> str:

        if NORMALIZE_LINE_ENDINGS:
            text = (
                text
                .replace("\r\n", "\n")
                .replace("\r", "\n")
            )

        if REMOVE_TRAILING_SPACES:
            lines = [
                line.rstrip()
                for line in text.split("\n")
            ]

            text = "\n".join(lines)

        if COLLAPSE_EXCESSIVE_BLANK_LINES:
            text = re.sub(
                r"\n{3,}",
                "\n\n",
                text,
            )

        return text.strip()