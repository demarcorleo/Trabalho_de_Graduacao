from extractors import (
    DoclingExtractor,
    MarkerExtractor,
    MarkItDownExtractor,
    MinerUExtractor,
    PDFMinerExtractor,
    PDFPlumberExtractor,
    PyMuPDF4LLMExtractor,
    PyMuPDFExtractor,
    PyPDFExtractor,
    UnstructuredExtractor,
)


class ExtractorFactory:

    _extractors = {
        "pymupdf": PyMuPDFExtractor,
        "pypdf": PyPDFExtractor,
        "pdfminer": PDFMinerExtractor,
        "pdfplumber": PDFPlumberExtractor,
        "pymupdf4llm": PyMuPDF4LLMExtractor,
        "docling": DoclingExtractor,
        "marker": MarkerExtractor,
        "markitdown": MarkItDownExtractor,
        "mineru": MinerUExtractor,
        "unstructured": UnstructuredExtractor,
    }

    @classmethod
    def create(cls, name: str):

        normalized_name = name.strip().lower()

        extractor_class = cls._extractors.get(
            normalized_name
        )

        if extractor_class is None:

            available = ", ".join(
                sorted(cls._extractors.keys())
            )

            raise ValueError(
                f"Extractor '{name}' não encontrado. "
                f"Disponíveis: {available}"
            )

        return extractor_class()

    @classmethod
    def available_extractors(cls):

        return sorted(
            cls._extractors.keys()
        )