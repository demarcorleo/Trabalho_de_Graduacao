from pathlib import Path
import subprocess

from base_extractor import BaseExtractor

from benchmark_config import (
    DOCLING_ENABLE_OCR,
    DOCLING_ENABLE_TABLE_STRUCTURE,
    MARKER_DISABLE_IMAGE_EXTRACTION,
    MARKER_DISABLE_OCR,
    MARKER_MODE,
    MARKER_OUTPUT_FORMAT,
    MARKER_USE_LLM,
    MARKITDOWN_ENABLE_PLUGINS,
    PDFMINER_CHAR_MARGIN,
    PDFMINER_LINE_MARGIN,
    PDFMINER_WORD_MARGIN,
    PDFPLUMBER_USE_TEXT_FLOW,
    PDFPLUMBER_X_TOLERANCE,
    PDFPLUMBER_Y_TOLERANCE,
    PYMUPDF_SORT_TEXT,
    TEST_SHELL_SCRIPT_PATH,
    TEST_SHELL_TIMEOUT_SECONDS,
    UNSTRUCTURED_STRATEGY,
)


# ============================================================
# PyMuPDF
# ============================================================

class PyMuPDFExtractor(BaseExtractor):

    name = "pymupdf"

    def extract(self, file_path: str) -> str:
        import pymupdf

        pages_text = []

        with pymupdf.open(file_path) as document:
            for page in document:
                page_text = page.get_text(
                    "text",
                    sort=PYMUPDF_SORT_TEXT,
                )
                pages_text.append(page_text or "")

        return "\n\n".join(pages_text)


# ============================================================
# pypdf
# ============================================================

class PyPDF2Extractor(BaseExtractor):

    name = "pypdf2"

    def extract(self, file_path: str) -> str:
        from pypdf import PdfReader

        reader = PdfReader(file_path)

        pages_text = []

        for page in reader.pages:
            page_text = page.extract_text()
            pages_text.append(page_text or "")

        return "\n\n".join(pages_text)


# ============================================================
# Docling
# ============================================================

class DoclingExtractor(BaseExtractor):

    name = "docling"

    def __init__(self):
        self.converter = None

    def setup(self) -> None:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import (
            DocumentConverter,
            PdfFormatOption,
        )

        pipeline_options = PdfPipelineOptions()

        pipeline_options.do_ocr = DOCLING_ENABLE_OCR

        pipeline_options.do_table_structure = (
            DOCLING_ENABLE_TABLE_STRUCTURE
        )

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

    def extract(self, file_path: str) -> str:
        if self.converter is None:
            raise RuntimeError(
                "DoclingExtractor.setup() não foi executado."
            )

        result = self.converter.convert(file_path)

        return result.document.export_to_text(
            page_break_placeholder="\n\n"
        )


# ============================================================
# Marker
# ============================================================

class MarkerExtractor(BaseExtractor):

    name = "marker"

    def __init__(self):
        self.converter = None
        self.text_from_rendered = None

    def setup(self) -> None:
        from marker.config.parser import ConfigParser
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered

        config = {
            "output_format": MARKER_OUTPUT_FORMAT,
            "mode": MARKER_MODE,
            "disable_ocr": MARKER_DISABLE_OCR,
            "disable_image_extraction": (
                MARKER_DISABLE_IMAGE_EXTRACTION
            ),
            "use_llm": MARKER_USE_LLM,
        }

        config_parser = ConfigParser(config)

        self.converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service(),
        )

        self.text_from_rendered = text_from_rendered

    def extract(self, file_path: str) -> str:
        if self.converter is None:
            raise RuntimeError(
                "MarkerExtractor.setup() não foi executado."
            )

        rendered = self.converter(file_path)

        text, _, _ = self.text_from_rendered(rendered)

        return text or ""


# ============================================================
# MarkItDown
# ============================================================

class MarkItDownExtractor(BaseExtractor):

    name = "markitdown"

    def __init__(self):
        self.converter = None

    def setup(self) -> None:
        from markitdown import MarkItDown

        self.converter = MarkItDown(
            enable_plugins=MARKITDOWN_ENABLE_PLUGINS
        )

    def extract(self, file_path: str) -> str:
        if self.converter is None:
            raise RuntimeError(
                "MarkItDownExtractor.setup() não foi executado."
            )

        result = self.converter.convert(file_path)

        return result.text_content or ""


# ============================================================
# Minecart
# ============================================================

class MinecartExtractor(BaseExtractor):

    name = "minecart"

    def extract(self, file_path: str) -> str:
        import minecart

        pages_text = []

        with open(file_path, "rb") as pdf_file:
            document = minecart.Document(pdf_file)

            for page in document.iter_pages():
                letterings = [
                    str(lettering)
                    for lettering in page.letterings
                ]

                pages_text.append("\n".join(letterings))

        return "\n\n".join(pages_text)


# ============================================================
# pdfminer.six
# ============================================================

class PDFMinerExtractor(BaseExtractor):

    name = "pdfminer"

    def extract(self, file_path: str) -> str:
        from pdfminer.high_level import extract_text
        from pdfminer.layout import LAParams

        laparams = LAParams(
            line_margin=PDFMINER_LINE_MARGIN,
            word_margin=PDFMINER_WORD_MARGIN,
            char_margin=PDFMINER_CHAR_MARGIN,
        )

        return extract_text(
            file_path,
            laparams=laparams,
        ) or ""


# ============================================================
# pdfplumber
# ============================================================

class PDFPlumberExtractor(BaseExtractor):

    name = "pdfplumber"

    def extract(self, file_path: str) -> str:
        import pdfplumber

        pages_text = []

        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text(
                    x_tolerance=PDFPLUMBER_X_TOLERANCE,
                    y_tolerance=PDFPLUMBER_Y_TOLERANCE,
                    use_text_flow=PDFPLUMBER_USE_TEXT_FLOW,
                )

                pages_text.append(page_text or "")

                # Evita acúmulo desnecessário de cache em PDFs maiores.
                page.close()

        return "\n\n".join(pages_text)


# ============================================================
# PyMuPDF4LLM
# ============================================================

class PyMuPDF4LLMExtractor(BaseExtractor):

    name = "pymupdf4llm"

    def extract(self, file_path: str) -> str:
        import pymupdf4llm

        return pymupdf4llm.to_text(file_path) or ""


# ============================================================
# Unstructured
# ============================================================

class UnstructuredExtractor(BaseExtractor):

    name = "unstructured"

    def extract(self, file_path: str) -> str:
        from unstructured.partition.pdf import partition_pdf

        elements = partition_pdf(
            filename=file_path,
            strategy=UNSTRUCTURED_STRATEGY,
        )

        text_elements = []

        for element in elements:
            element_text = str(element).strip()

            if element_text:
                text_elements.append(element_text)

        return "\n\n".join(text_elements)


# ============================================================
# OCRmyPDF
# ============================================================

class OCRmyPDFExtractor(BaseExtractor):

    name = "ocrmypdf"

    def extract(self, file_path: str) -> str:
        raise RuntimeError(
            "OCRmyPDF foi excluído do benchmark textual principal. "
            "Ele adiciona uma camada OCR ao PDF e não realiza a mesma "
            "tarefa dos extratores de camada textual."
        )


# ============================================================
# pytesseract
# ============================================================

class PyTesseractExtractor(BaseExtractor):

    name = "pytesseract"

    def extract(self, file_path: str) -> str:
        raise RuntimeError(
            "pytesseract foi excluído do benchmark textual principal. "
            "Ele exige renderização das páginas como imagens e OCR, "
            "o que não é equivalente à extração da camada textual."
        )


# ============================================================
# test.sh
# ============================================================

class TestShellExtractor(BaseExtractor):

    name = "test.sh"

    def extract(self, file_path: str) -> str:
        if TEST_SHELL_SCRIPT_PATH is None:
            raise RuntimeError(
                "TEST_SHELL_SCRIPT_PATH ainda não foi configurado "
                "em benchmark_config.py."
            )

        script_path = Path(TEST_SHELL_SCRIPT_PATH)

        if not script_path.exists():
            raise FileNotFoundError(
                f"Script não encontrado: {script_path}"
            )

        process = subprocess.run(
            [
                "bash",
                str(script_path),
                file_path,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=TEST_SHELL_TIMEOUT_SECONDS,
        )

        return process.stdout or ""