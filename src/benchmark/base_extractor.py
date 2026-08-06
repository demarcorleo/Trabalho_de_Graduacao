from abc import ABC, abstractmethod
import re

from benchmark_config import (
    COLLAPSE_EXCESSIVE_BLANK_LINES,
    NORMALIZE_LINE_ENDINGS,
    REMOVE_TRAILING_SPACES,
)


class BaseExtractor(ABC):
    """
    Contrato comum para todas as bibliotecas avaliadas.

    Todos os extratores devem:

    1. receber o caminho de um PDF pesquisável;
    2. extrair somente o conteúdo textual;
    3. retornar uma string;
    4. não salvar arquivos internamente;
    5. não executar OCR no benchmark principal;
    6. não utilizar LLM;
    7. não aplicar pós-processamento exclusivo.
    """

    name: str = "base"

    def setup(self) -> None:
        """
        Inicializações realizadas uma única vez.

        Modelos, conversores e objetos pesados devem ser carregados aqui,
        e não dentro do tempo de extração de cada documento.
        """

    def teardown(self) -> None:
        """
        Liberação opcional de recursos após o benchmark.
        """

    @abstractmethod
    def extract(self, file_path: str) -> str:
        """
        Extrai o texto do documento e retorna uma string.
        """
        raise NotImplementedError

    def run(self, file_path: str) -> str:
        """
        Executa o extrator e aplica a mesma normalização mínima
        para todas as bibliotecas.
        """
        text = self.extract(file_path)

        if text is None:
            text = ""

        if not isinstance(text, str):
            text = str(text)

        return self.normalize_text(text)

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalização mínima e uniforme.

        Não tenta corrigir:
        - ordem de leitura;
        - títulos;
        - tabelas;
        - cabeçalhos;
        - rodapés;
        - caracteres extraídos incorretamente.

        Esses elementos fazem parte do resultado avaliado.
        """

        if NORMALIZE_LINE_ENDINGS:
            text = text.replace("\r\n", "\n").replace("\r", "\n")

        if REMOVE_TRAILING_SPACES:
            lines = [line.rstrip() for line in text.split("\n")]
            text = "\n".join(lines)

        if COLLAPSE_EXCESSIVE_BLANK_LINES:
            text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()