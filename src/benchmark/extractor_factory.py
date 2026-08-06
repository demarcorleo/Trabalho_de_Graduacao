from extractors import (
    DoclingExtractor,
    MarkerExtractor,
    MarkItDownExtractor,
    MinecartExtractor,
    OCRmyPDFExtractor,
    PDFMinerExtractor,
    PDFPlumberExtractor,
    PyMuPDF4LLMExtractor,
    PyMuPDFExtractor,
    PyPDF2Extractor,
    PyTesseractExtractor,
    TestShellExtractor,
    UnstructuredExtractor,
)


class ExtractorFactory:

    _extractors = {
        "pymupdf": PyMuPDFExtractor,
        "pypdf2": PyPDF2Extractor,
        "pypdf": PyPDF2Extractor,
        "docling": DoclingExtractor,
        "marker": MarkerExtractor,
        "markitdown": MarkItDownExtractor,
        "minecart": MinecartExtractor,
        "ocrmypdf": OCRmyPDFExtractor,
        "pdfminer": PDFMinerExtractor,
        "pdfminer.six": PDFMinerExtractor,
        "pdfplumber": PDFPlumberExtractor,
        "pymupdf4llm": PyMuPDF4LLMExtractor,
        "pytesseract": PyTesseractExtractor,
        "test.sh": TestShellExtractor,
        "unstructured": UnstructuredExtractor,
    }

    @classmethod
    def create(cls, name: str):
        normalized_name = name.strip().lower()

        extractor_class = cls._extractors.get(normalized_name)

        if extractor_class is None:
            available = ", ".join(sorted(cls._extractors))

            raise ValueError(
                f"Extractor '{name}' não encontrado. "
                f"Opções disponíveis: {available}"
            )

        return extractor_class()

    @classmethod
    def available_extractors(cls) -> list[str]:
        return sorted(cls._extractors)