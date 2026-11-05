import json
import re
from pathlib import Path

from config.settings import DOCUMENTS_PATH

from src.ingestion.ingestion import carregar_documentos
from src.ingestion.generate_chunking import gerar_chunks


QUESTIONS_PATH = Path(
    "src/evaluation/questions.json"
)


QUESTOES_PROBLEMATICAS = {
    "q04": {
        "termos": [
            "analog",
            "output",
            "4",
            "20",
            "10",
        ]
    },
    "q05": {
        "termos": [
            "0",
            "2",
            "bar",
            "B3",
        ]
    },
    "q07": {
        "termos": [
            "pH",
            "display",
            "range",
        ]
    },
    "q08": {
        "termos": [
            "C",
            "0.1",
            "2000",
            "µS",
        ]
    },
    "q12": {
        "termos": [
            "flow",
            "direction",
            "vertically",
            "upwards",
        ]
    },
}


def normalizar(texto):
    if texto is None:
        return ""

    texto = str(texto).lower()

    texto = texto.replace(
        "\u2212",
        "-"
    )

    texto = texto.replace(
        "\u00a0",
        " "
    )

    texto = re.sub(
        r"\s+",
        " ",
        texto
    )

    return texto.strip()


def carregar_questoes():
    with open(
        QUESTIONS_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def encontrar_documento(
    documentos,
    source
):
    source = Path(
        source
    ).name

    for documento in documentos:
        doc_source = Path(
            str(
                documento.metadata.get(
                    "source",
                    ""
                )
            )
        ).name

        if doc_source == source:
            return documento

    return None


def procurar_termos(
    texto,
    termos
):
    texto_normalizado = normalizar(
        texto
    )

    encontrados = []

    for termo in termos:
        if normalizar(termo) in texto_normalizado:
            encontrados.append(termo)

    return encontrados


def imprimir_trecho(
    texto,
    termo,
    janela=500
):
    texto_normalizado = normalizar(
        texto
    )

    termo_normalizado = normalizar(
        termo
    )

    posicao = texto_normalizado.find(
        termo_normalizado
    )

    if posicao == -1:
        return

    inicio = max(
        0,
        posicao - janela
    )

    fim = min(
        len(texto),
        posicao + len(termo) + janela
    )

    print(
        texto[inicio:fim]
    )


def main():

    print("=" * 100)
    print(
        "DIAGNÓSTICO DOS GOLD SOURCES"
    )
    print("=" * 100)

    print(
        "\nCarregando questões..."
    )

    questions = carregar_questoes()

    print(
        f"Questões carregadas: "
        f"{len(questions)}"
    )

    print(
        "\nCarregando documentos..."
    )

    documentos = carregar_documentos(
        DOCUMENTS_PATH
    )

    print(
        f"Documentos carregados: "
        f"{len(documentos)}"
    )

    print(
        "\nGerando chunks C3 "
        "(1000 / 150)..."
    )

    chunks = gerar_chunks(
        documentos
    )

    print(
        f"Chunks gerados: "
        f"{len(chunks)}"
    )

    # ------------------------------------------------------------
    # QUESTÕES PROBLEMÁTICAS
    # ------------------------------------------------------------

    for question in questions:

        question_id = question["id"]

        if (
            question_id
            not in QUESTOES_PROBLEMATICAS
        ):
            continue

        source = question[
            "source"
        ]["doc"]

        gold = question[
            "gold"
        ]

        gold_source = question[
            "gold_source"
        ]

        termos = (
            QUESTOES_PROBLEMATICAS[
                question_id
            ]["termos"]
        )

        print("\n")
        print("=" * 100)
        print(
            f"{question_id} | "
            f"{question['question']}"
        )
        print("=" * 100)

        print(
            f"\nDocumento esperado:\n"
            f"{source}"
        )

        print(
            f"\nGold:\n"
            f"{gold}"
        )

        print(
            f"\nGold source atual:\n"
            f"{gold_source}"
        )

        print(
            "\nTermos procurados:"
        )

        print(
            ", ".join(termos)
        )

        # --------------------------------------------------------
        # DOCUMENTO
        # --------------------------------------------------------

        documento = encontrar_documento(
            documentos,
            source
        )

        if documento is None:

            print(
                "\nERRO: documento não "
                "encontrado."
            )

            continue

        texto_documento = (
            documento.get_content()
        )

        print(
            "\n" + "-" * 100
        )

        print(
            "TERMOS ENCONTRADOS "
            "NO DOCUMENTO ORIGINAL"
        )

        print(
            "-" * 100
        )

        encontrados_documento = (
            procurar_termos(
                texto_documento,
                termos
            )
        )

        print(
            f"\nEncontrados: "
            f"{encontrados_documento}"
        )

        for termo in encontrados_documento:

            print(
                "\n" + "-" * 80
            )

            print(
                f"Contexto do termo: "
                f"{termo}"
            )

            print(
                "-" * 80
            )

            imprimir_trecho(
                texto_documento,
                termo
            )

        # --------------------------------------------------------
        # CHUNKS
        # --------------------------------------------------------

        print(
            "\n" + "-" * 100
        )

        print(
            "CHUNKS DO DOCUMENTO "
            "QUE CONTÊM OS TERMOS"
        )

        print(
            "-" * 100
        )

        chunks_encontrados = []

        for indice, chunk in enumerate(
            chunks
        ):

            chunk_source = Path(
                str(
                    chunk.metadata.get(
                        "source",
                        ""
                    )
                )
            ).name

            if chunk_source != Path(
                source
            ).name:
                continue

            texto = chunk.text

            encontrados = (
                procurar_termos(
                    texto,
                    termos
                )
            )

            if encontrados:

                chunks_encontrados.append(
                    indice
                )

                print(
                    "\n"
                    + "=" * 100
                )

                print(
                    f"CHUNK INDEX: {indice}"
                )

                print(
                    f"TERMOS: "
                    f"{encontrados}"
                )

                print(
                    "=" * 100
                )

                print(texto)

        print(
            "\n" + "-" * 100
        )

        print(
            "RESUMO"
        )

        print(
            "-" * 100
        )

        print(
            f"Chunks encontrados: "
            f"{chunks_encontrados}"
        )

        print(
            f"Quantidade: "
            f"{len(chunks_encontrados)}"
        )

    print("\n")
    print("=" * 100)
    print(
        "DIAGNÓSTICO FINALIZADO"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()