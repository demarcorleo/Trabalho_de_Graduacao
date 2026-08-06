import csv
import gc
import statistics
import time
import traceback
from pathlib import Path

import pymupdf

from benchmark_config import (
    DATASET_DIR,
    ENABLED_EXTRACTORS,
    ERROR_LOG_PATH,
    EXTRACTED_TEXT_DIR,
    FAIL_FAST,
    MEASURED_RUNS,
    OUTPUT_ENCODING,
    OUTPUT_EXTENSION,
    PDF_PATTERN,
    RESULTS_CSV_PATH,
    RESULTS_DIR,
    SAVE_EXTRACTED_TEXT,
    SAVE_ONLY_FIRST_MEASURED_RUN,
    SEARCH_RECURSIVELY,
    SUMMARY_CSV_PATH,
    WARMUP_RUNS,
)
from extractor_factory import ExtractorFactory


# ============================================================
# DATASET
# ============================================================

def listar_pdfs() -> list[Path]:
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Pasta do dataset não encontrada: {DATASET_DIR}"
        )

    if SEARCH_RECURSIVELY:
        files = list(DATASET_DIR.rglob(PDF_PATTERN))
    else:
        files = list(DATASET_DIR.glob(PDF_PATTERN))

    files = sorted(
        file
        for file in files
        if file.is_file()
    )

    if not files:
        raise FileNotFoundError(
            f"Nenhum PDF foi encontrado em: {DATASET_DIR}"
        )

    return files


# ============================================================
# METADADOS INDEPENDENTES
# ============================================================

def obter_metadados_pdf(file_path: Path) -> dict:
    """
    Mede características do arquivo uma única vez.

    Essa rotina não faz parte do tempo de nenhuma biblioteca.
    """

    file_size_bytes = file_path.stat().st_size

    with pymupdf.open(file_path) as document:
        page_count = len(document)

    return {
        "arquivo": file_path.name,
        "caminho": str(file_path),
        "tamanho_bytes": file_size_bytes,
        "paginas": page_count,
    }


# ============================================================
# SAÍDA
# ============================================================

def preparar_diretorios() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    EXTRACTED_TEXT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def salvar_texto(
    library_name: str,
    pdf_path: Path,
    text: str,
) -> Path:
    library_dir = EXTRACTED_TEXT_DIR / library_name

    library_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        library_dir
        / f"{pdf_path.stem}{OUTPUT_EXTENSION}"
    )

    output_path.write_text(
        text,
        encoding=OUTPUT_ENCODING,
    )

    return output_path


def salvar_resultados_csv(results: list[dict]) -> None:
    if not results:
        return

    fieldnames = [
        "biblioteca",
        "arquivo",
        "caminho",
        "repeticao",
        "status",
        "tempo_segundos",
        "paginas",
        "tamanho_bytes",
        "caracteres",
        "palavras",
        "linhas",
        "arquivo_saida",
        "erro_tipo",
        "erro_mensagem",
    ]

    with RESULTS_CSV_PATH.open(
        "w",
        newline="",
        encoding=OUTPUT_ENCODING,
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(results)


def salvar_resumo_csv(results: list[dict]) -> None:
    successful_results = [
        result
        for result in results
        if result["status"] == "sucesso"
    ]

    grouped = {}

    for result in successful_results:
        key = result["biblioteca"]

        grouped.setdefault(key, []).append(result)

    summary_rows = []

    for library_name, library_results in grouped.items():
        times = [
            result["tempo_segundos"]
            for result in library_results
        ]

        characters = [
            result["caracteres"]
            for result in library_results
        ]

        total_pages = sum(
            result["paginas"]
            for result in library_results
        )

        total_time = sum(times)

        pages_per_second = (
            total_pages / total_time
            if total_time > 0
            else 0.0
        )

        summary_rows.append({
            "biblioteca": library_name,
            "execucoes_validas": len(library_results),
            "tempo_total_segundos": total_time,
            "tempo_medio_segundos": statistics.mean(times),
            "tempo_mediano_segundos": statistics.median(times),
            "tempo_minimo_segundos": min(times),
            "tempo_maximo_segundos": max(times),
            "desvio_padrao_segundos": (
                statistics.stdev(times)
                if len(times) > 1
                else 0.0
            ),
            "paginas_processadas": total_pages,
            "paginas_por_segundo": pages_per_second,
            "caracteres_medio": statistics.mean(characters),
        })

    fieldnames = [
        "biblioteca",
        "execucoes_validas",
        "tempo_total_segundos",
        "tempo_medio_segundos",
        "tempo_mediano_segundos",
        "tempo_minimo_segundos",
        "tempo_maximo_segundos",
        "desvio_padrao_segundos",
        "paginas_processadas",
        "paginas_por_segundo",
        "caracteres_medio",
    ]

    with SUMMARY_CSV_PATH.open(
        "w",
        newline="",
        encoding=OUTPUT_ENCODING,
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(summary_rows)


def registrar_erro(
    library_name: str,
    pdf_path: Path | None,
    error: Exception,
) -> None:
    file_name = (
        pdf_path.name
        if pdf_path is not None
        else "setup"
    )

    error_text = (
        "\n"
        + "=" * 80
        + "\n"
        + f"Biblioteca: {library_name}\n"
        + f"Arquivo: {file_name}\n"
        + f"Tipo: {type(error).__name__}\n"
        + f"Mensagem: {error}\n\n"
        + traceback.format_exc()
        + "\n"
    )

    with ERROR_LOG_PATH.open(
        "a",
        encoding=OUTPUT_ENCODING,
    ) as error_file:
        error_file.write(error_text)


# ============================================================
# MÉTRICAS TEXTUAIS
# ============================================================

def calcular_metricas_texto(text: str) -> dict:
    return {
        "caracteres": len(text),
        "palavras": len(text.split()),
        "linhas": len(text.splitlines()),
    }


# ============================================================
# EXECUÇÃO DE UMA BIBLIOTECA
# ============================================================

def executar_biblioteca(
    library_name: str,
    pdf_files: list[Path],
    pdf_metadata: dict[str, dict],
) -> list[dict]:

    results = []

    print("\n" + "=" * 70)
    print(f"Biblioteca: {library_name}")
    print("=" * 70)

    extractor = ExtractorFactory.create(library_name)

    try:
        print("Inicializando biblioteca...")
        extractor.setup()
        print("Inicialização concluída.")

    except Exception as error:
        print(f"Erro na inicialização de {library_name}: {error}")

        registrar_erro(
            library_name,
            None,
            error,
        )

        if FAIL_FAST:
            raise

        return results

    try:
        # ----------------------------------------------------
        # WARM-UP
        # ----------------------------------------------------

        if WARMUP_RUNS > 0:
            print(f"Warm-up: {WARMUP_RUNS} execução(ões)")

            warmup_file = pdf_files[0]

            for _ in range(WARMUP_RUNS):
                extractor.run(str(warmup_file))

        # ----------------------------------------------------
        # EXECUÇÕES MEDIDAS
        # ----------------------------------------------------

        for file_index, pdf_path in enumerate(pdf_files, start=1):
            metadata = pdf_metadata[str(pdf_path)]

            print(
                f"\n[{file_index}/{len(pdf_files)}] "
                f"{pdf_path.name}"
            )

            for repetition in range(1, MEASURED_RUNS + 1):
                print(
                    f"  Repetição "
                    f"{repetition}/{MEASURED_RUNS}",
                    end="",
                    flush=True,
                )

                output_path = ""

                try:
                    gc.collect()

                    start_time = time.perf_counter()

                    text = extractor.run(str(pdf_path))

                    elapsed_time = (
                        time.perf_counter() - start_time
                    )

                    text_metrics = calcular_metricas_texto(text)

                    should_save = (
                        SAVE_EXTRACTED_TEXT
                        and (
                            not SAVE_ONLY_FIRST_MEASURED_RUN
                            or repetition == 1
                        )
                    )

                    if should_save:
                        saved_path = salvar_texto(
                            library_name,
                            pdf_path,
                            text,
                        )

                        output_path = str(saved_path)

                    result = {
                        "biblioteca": library_name,
                        "arquivo": pdf_path.name,
                        "caminho": str(pdf_path),
                        "repeticao": repetition,
                        "status": "sucesso",
                        "tempo_segundos": elapsed_time,
                        "paginas": metadata["paginas"],
                        "tamanho_bytes": metadata["tamanho_bytes"],
                        "caracteres": text_metrics["caracteres"],
                        "palavras": text_metrics["palavras"],
                        "linhas": text_metrics["linhas"],
                        "arquivo_saida": output_path,
                        "erro_tipo": "",
                        "erro_mensagem": "",
                    }

                    results.append(result)

                    print(
                        f" — {elapsed_time:.4f} s "
                        f"— {text_metrics['caracteres']} caracteres"
                    )

                except Exception as error:
                    print(f" — ERRO: {error}")

                    registrar_erro(
                        library_name,
                        pdf_path,
                        error,
                    )

                    result = {
                        "biblioteca": library_name,
                        "arquivo": pdf_path.name,
                        "caminho": str(pdf_path),
                        "repeticao": repetition,
                        "status": "erro",
                        "tempo_segundos": "",
                        "paginas": metadata["paginas"],
                        "tamanho_bytes": metadata["tamanho_bytes"],
                        "caracteres": "",
                        "palavras": "",
                        "linhas": "",
                        "arquivo_saida": "",
                        "erro_tipo": type(error).__name__,
                        "erro_mensagem": str(error),
                    }

                    results.append(result)

                    if FAIL_FAST:
                        raise

    finally:
        extractor.teardown()

    return results


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    preparar_diretorios()

    # Limpa o log da execução anterior.
    ERROR_LOG_PATH.write_text(
        "",
        encoding=OUTPUT_ENCODING,
    )

    pdf_files = listar_pdfs()

    print("\nBENCHMARK DE EXTRAÇÃO TEXTUAL DE PDFs")
    print("=" * 70)
    print(f"Dataset: {DATASET_DIR}")
    print(f"PDFs encontrados: {len(pdf_files)}")
    print(f"Bibliotecas: {len(ENABLED_EXTRACTORS)}")
    print(f"Warm-ups: {WARMUP_RUNS}")
    print(f"Repetições medidas: {MEASURED_RUNS}")
    print(f"Saída: TXT UTF-8")

    pdf_metadata = {
        str(pdf_path): obter_metadados_pdf(pdf_path)
        for pdf_path in pdf_files
    }

    all_results = []

    for library_name in ENABLED_EXTRACTORS:
        library_results = executar_biblioteca(
            library_name,
            pdf_files,
            pdf_metadata,
        )

        all_results.extend(library_results)

        # Salva também durante a execução.
        # Assim, resultados anteriores não são perdidos
        # caso uma biblioteca posterior interrompa o processo.
        salvar_resultados_csv(all_results)

        salvar_resumo_csv(all_results)

    print("\n" + "=" * 70)
    print("Benchmark concluído.")
    print(f"Resultados detalhados: {RESULTS_CSV_PATH}")
    print(f"Resumo: {SUMMARY_CSV_PATH}")
    print(f"Textos extraídos: {EXTRACTED_TEXT_DIR}")
    print(f"Log de erros: {ERROR_LOG_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()