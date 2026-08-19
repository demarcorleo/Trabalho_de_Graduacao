from importlib.util import find_spec
from pathlib import Path
import subprocess
import tempfile
import time
import urllib.request

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

    MINERU_API_HOST,
    MINERU_API_PORT,
    MINERU_API_STARTUP_TIMEOUT_SECONDS,
    MINERU_BACKEND,
    MINERU_ENABLE_FORMULA,
    MINERU_ENABLE_IMAGE_ANALYSIS,
    MINERU_ENABLE_TABLE,
    MINERU_METHOD,
    MINERU_PROCESS_TIMEOUT_SECONDS,
    MINERU_SERVER_LOG_PATH,

    PDFMINER_CHAR_MARGIN,
    PDFMINER_LINE_MARGIN,
    PDFMINER_WORD_MARGIN,

    PDFPLUMBER_USE_TEXT_FLOW,
    PDFPLUMBER_X_TOLERANCE,
    PDFPLUMBER_Y_TOLERANCE,

    PYMUPDF_SORT_TEXT,

    UNSTRUCTURED_STRATEGY,
    UNSTRUCTURED_LANGUAGES,
)


# ============================================================
# AUXILIAR
# ============================================================

def verificar_modulo(
    module_name: str,
    package_name: str | None = None,
):

    if find_spec(module_name) is None:

        package = (
            package_name
            if package_name
            else module_name
        )

        raise RuntimeError(
            f"A dependência '{package}' "
            "não está instalada."
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

                pages_text.append(
                    page_text or ""
                )

        return "\n\n".join(
            pages_text
        )


# ============================================================
# pypdf
# ============================================================

class PyPDFExtractor(BaseExtractor):

    name = "pypdf"

    def setup(self) -> None:

        verificar_modulo(
            "pypdf",
            "pypdf",
        )

    def extract(self, file_path: str) -> str:

        from pypdf import PdfReader

        reader = PdfReader(file_path)

        pages_text = []

        for page in reader.pages:

            page_text = page.extract_text()

            pages_text.append(
                page_text or ""
            )

        return "\n\n".join(
            pages_text
        )


# ============================================================
# PDFMiner
# ============================================================

class PDFMinerExtractor(BaseExtractor):

    name = "pdfminer"

    def setup(self) -> None:

        verificar_modulo(
            "pdfminer.high_level",
            "pdfminer.six",
        )

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

    def setup(self) -> None:

        verificar_modulo(
            "pdfplumber",
        )

    def extract(self, file_path: str) -> str:

        import pdfplumber

        pages_text = []

        with pdfplumber.open(
            file_path
        ) as pdf:

            for page in pdf.pages:

                page_text = (
                    page.extract_text(
                        x_tolerance=(
                            PDFPLUMBER_X_TOLERANCE
                        ),
                        y_tolerance=(
                            PDFPLUMBER_Y_TOLERANCE
                        ),
                        use_text_flow=(
                            PDFPLUMBER_USE_TEXT_FLOW
                        ),
                    )
                )

                pages_text.append(
                    page_text or ""
                )

                page.close()

        return "\n\n".join(
            pages_text
        )


# ============================================================
# PyMuPDF4LLM
# ============================================================

class PyMuPDF4LLMExtractor(BaseExtractor):

    name = "pymupdf4llm"

    def setup(self) -> None:

        verificar_modulo(
            "pymupdf4llm",
        )

    def extract(self, file_path: str) -> str:

        import pymupdf4llm

        return (
            pymupdf4llm.to_text(
                file_path
            )
            or ""
        )


# ============================================================
# Docling
# ============================================================

class DoclingExtractor(BaseExtractor):

    name = "docling"

    def __init__(self):

        self.converter = None

    def setup(self) -> None:

        from docling.datamodel.base_models import (
            InputFormat,
        )

        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
        )

        from docling.document_converter import (
            DocumentConverter,
            PdfFormatOption,
        )

        pipeline_options = (
            PdfPipelineOptions()
        )

        pipeline_options.do_ocr = (
            DOCLING_ENABLE_OCR
        )

        pipeline_options.do_table_structure = (
            DOCLING_ENABLE_TABLE_STRUCTURE
        )

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF:
                    PdfFormatOption(
                        pipeline_options=(
                            pipeline_options
                        )
                    )
            }
        )

    def extract(self, file_path: str) -> str:

        if self.converter is None:

            raise RuntimeError(
                "Docling não inicializado."
            )

        result = self.converter.convert(
            file_path
        )

        return (
            result.document.export_to_text(
                page_break_placeholder="\n\n"
            )
        )


# ============================================================
# Marker
# marker-pdf 2.0.0
# ============================================================

class MarkerExtractor(BaseExtractor):

    name = "marker"

    def __init__(self):

        self.converter = None

        self.text_from_rendered = None

    def setup(self) -> None:

        from marker.config.parser import (
            ConfigParser,
        )

        from marker.converters.pdf import (
            PdfConverter,
        )

        from marker.models import (
            create_model_dict,
        )

        from marker.output import (
            text_from_rendered,
        )

        config = {
            "output_format":
                MARKER_OUTPUT_FORMAT,

            "mode":
                MARKER_MODE,

            "disable_ocr":
                MARKER_DISABLE_OCR,

            "disable_image_extraction":
                MARKER_DISABLE_IMAGE_EXTRACTION,

            "use_llm":
                MARKER_USE_LLM,
        }

        config_parser = ConfigParser(
            config
        )

        converter_kwargs = {
            "config":
                config_parser.generate_config_dict(),

            "artifact_dict":
                create_model_dict(),

            "processor_list":
                config_parser.get_processors(),

            "renderer":
                config_parser.get_renderer(),
        }

        if MARKER_USE_LLM:

            converter_kwargs[
                "llm_service"
            ] = (
                config_parser
                .get_llm_service()
            )

        self.converter = PdfConverter(
            **converter_kwargs
        )

        self.text_from_rendered = (
            text_from_rendered
        )

    def extract(self, file_path: str) -> str:

        if self.converter is None:

            raise RuntimeError(
                "Marker não inicializado."
            )

        rendered = self.converter(
            file_path
        )

        text, _, _ = (
            self.text_from_rendered(
                rendered
            )
        )

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
            enable_plugins=(
                MARKITDOWN_ENABLE_PLUGINS
            )
        )

    def extract(self, file_path: str) -> str:

        if self.converter is None:

            raise RuntimeError(
                "MarkItDown não inicializado."
            )

        result = self.converter.convert(
            file_path
        )

        return (
            result.text_content
            or ""
        )


# ============================================================
# Unstructured
# ============================================================

class UnstructuredExtractor(BaseExtractor):

    name = "unstructured"

    def setup(self) -> None:

        verificar_modulo(
            "unstructured",
            "unstructured",
        )

    def extract(self, file_path: str) -> str:

        from unstructured.partition.pdf import (
            partition_pdf,
        )

        elements = partition_pdf(
            filename=file_path,
            strategy=UNSTRUCTURED_STRATEGY,
            languages=UNSTRUCTURED_LANGUAGES,
        )

        text_elements = []

        for element in elements:

            text = str(
                element
            ).strip()

            if text:

                text_elements.append(
                    text
                )

        return "\n\n".join(
            text_elements
        )


# ============================================================
# MinerU
# ============================================================

class MinerUExtractor(BaseExtractor):

    name = "mineru"

    def __init__(self):

        self.api_process = None

        self.log_file = None

    @property
    def api_url(self):

        return (
            f"http://"
            f"{MINERU_API_HOST}:"
            f"{MINERU_API_PORT}"
        )

    def setup(self) -> None:

        verificar_modulo(
            "mineru",
            "mineru",
        )

        MINERU_SERVER_LOG_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.log_file = open(
            MINERU_SERVER_LOG_PATH,
            "w",
            encoding="utf-8",
        )

        command = [
            "mineru-api",
            "--host",
            MINERU_API_HOST,
            "--port",
            str(MINERU_API_PORT),
        ]

        self.api_process = subprocess.Popen(
            command,
            stdout=self.log_file,
            stderr=self.log_file,
        )

        deadline = (
            time.time()
            + MINERU_API_STARTUP_TIMEOUT_SECONDS
        )

        health_url = (
            self.api_url
            + "/health"
        )

        while time.time() < deadline:

            if (
                self.api_process.poll()
                is not None
            ):
                raise RuntimeError(
                    "MinerU API encerrou durante "
                    "a inicialização. Consulte: "
                    f"{MINERU_SERVER_LOG_PATH}"
                )

            try:

                with urllib.request.urlopen(
                    health_url,
                    timeout=2,
                ) as response:

                    if response.status == 200:
                        return

            except Exception:
                pass

            time.sleep(1)

        raise TimeoutError(
            "MinerU API não ficou pronta dentro "
            "do tempo configurado. Consulte: "
            f"{MINERU_SERVER_LOG_PATH}"
        )

    def extract(self, file_path: str) -> str:

        with tempfile.TemporaryDirectory(
            prefix="mineru_benchmark_"
        ) as temp_dir:

            command = [
                "mineru",

                "-p",
                str(file_path),

                "-o",
                temp_dir,

                "--api-url",
                self.api_url,

                "-m",
                MINERU_METHOD,

                "-b",
                MINERU_BACKEND,

                "--formula",
                str(
                    MINERU_ENABLE_FORMULA
                ).lower(),

                "--table",
                str(
                    MINERU_ENABLE_TABLE
                ).lower(),

                "--image-analysis",
                str(
                    MINERU_ENABLE_IMAGE_ANALYSIS
                ).lower(),
            ]

            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=(
                    MINERU_PROCESS_TIMEOUT_SECONDS
                ),
            )

            if process.returncode != 0:

                raise RuntimeError(
                    "MinerU falhou.\n"
                    f"STDOUT:\n{process.stdout}\n\n"
                    f"STDERR:\n{process.stderr}"
                )

            temp_path = Path(
                temp_dir
            )

            markdown_files = list(
                temp_path.rglob("*.md")
            )

            if not markdown_files:

                raise RuntimeError(
                    "MinerU concluiu a execução, "
                    "mas nenhum arquivo Markdown "
                    "foi encontrado no diretório "
                    "de saída."
                )

            main_markdown = max(
                markdown_files,
                key=lambda path:
                    path.stat().st_size,
            )

            return main_markdown.read_text(
                encoding="utf-8",
                errors="replace",
            )

    def teardown(self) -> None:

        if self.api_process is not None:

            if (
                self.api_process.poll()
                is None
            ):

                self.api_process.terminate()

                try:

                    self.api_process.wait(
                        timeout=10
                    )

                except subprocess.TimeoutExpired:

                    self.api_process.kill()

                    self.api_process.wait()

        if self.log_file is not None:

            self.log_file.close()