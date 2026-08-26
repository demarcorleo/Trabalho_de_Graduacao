import csv
import gc
import multiprocessing as mp
import statistics
import tempfile
import time
import traceback
import uuid
from pathlib import Path

import pymupdf

from benchmark_config import (
    DATASET_DIR,
    ENABLED_EXTRACTORS,
    ERROR_LOG_PATH,
    EXTRACTED_TEXT_DIR,
    FAIL_FAST,
    INCLUDED_DOCUMENTS,
    MEASURED_RUNS,
    OUTPUT_ENCODING,
    OUTPUT_EXTENSION,
    PAGE_SELECTION,
    PDF_PATTERN,
    RESULTS_CSV_PATH,
    RESULTS_DIR,
    SAVE_EXTRACTED_TEXT,
    SAVE_ONLY_FIRST_MEASURED_RUN,
    SEARCH_RECURSIVELY,
    SUMMARY_CSV_PATH,
    WARMUP_RUNS,
)

from extractor_factory import (
    ExtractorFactory,
)

from resource_monitor import (
    ResourceMonitor,
)


# ============================================================
# CONFIGURAÇÃO DO PROCESSO ISOLADO
# ============================================================

PROCESS_TIMEOUT_SECONDS = 300


# ============================================================
# DATASET
# ============================================================

def listar_pdfs():

    if not DATASET_DIR.exists():

        raise FileNotFoundError(
            f"Dataset não encontrado: "
            f"{DATASET_DIR}"
        )

    if SEARCH_RECURSIVELY:

        files = list(
            DATASET_DIR.rglob(
                PDF_PATTERN
            )
        )

    else:

        files = list(
            DATASET_DIR.glob(
                PDF_PATTERN
            )
        )

    files = sorted(
        file
        for file in files
        if file.is_file()
    )

    if INCLUDED_DOCUMENTS is not None:

        included = set(
            INCLUDED_DOCUMENTS
        )

        files = [
            file
            for file in files
            if file.name in included
        ]

        encontrados = {
            file.name
            for file in files
        }

        ausentes = (
            included
            - encontrados
        )

        if ausentes:

            nomes = ", ".join(
                sorted(ausentes)
            )

            raise FileNotFoundError(
                "Documentos configurados em "
                "INCLUDED_DOCUMENTS não encontrados: "
                f"{nomes}"
            )

    if not files:

        raise FileNotFoundError(
            "Nenhum PDF foi selecionado."
        )

    return files


# ============================================================
# PÁGINAS
# ============================================================

def validar_paginas(
    pdf_path: Path,
    paginas: list[int],
):

    if not paginas:

        raise ValueError(
            f"A seleção de páginas de "
            f"'{pdf_path.name}' está vazia."
        )

    if any(
        not isinstance(
            pagina,
            int
        )
        for pagina in paginas
    ):

        raise TypeError(
            "As páginas devem ser inteiros."
        )

    if any(
        pagina < 1
        for pagina in paginas
    ):

        raise ValueError(
            "A numeração das páginas começa em 1."
        )

    if len(paginas) != len(
        set(paginas)
    ):

        raise ValueError(
            f"Há páginas repetidas em "
            f"'{pdf_path.name}'."
        )

    with pymupdf.open(
        pdf_path
    ) as document:

        total_pages = len(
            document
        )

    invalidas = [
        pagina
        for pagina in paginas
        if pagina > total_pages
    ]

    if invalidas:

        raise ValueError(
            f"'{pdf_path.name}' possui "
            f"{total_pages} páginas. "
            f"Páginas inválidas: {invalidas}"
        )

    return total_pages


def criar_pdf_selecionado(
    original_path: Path,
    paginas: list[int],
    output_path: Path,
):

    validar_paginas(
        original_path,
        paginas,
    )

    with pymupdf.open(
        original_path
    ) as original:

        novo = pymupdf.open()

        for pagina in paginas:

            indice = pagina - 1

            novo.insert_pdf(
                original,
                from_page=indice,
                to_page=indice,
            )

        novo.save(
            output_path
        )

        novo.close()


def preparar_documentos(
    pdf_files,
    temporary_dir: Path,
):

    documentos = []

    for original_path in pdf_files:

        with pymupdf.open(
            original_path
        ) as document:

            total_original_pages = len(
                document
            )

        paginas_configuradas = (
            PAGE_SELECTION.get(
                original_path.name
            )
        )

        if paginas_configuradas is None:

            input_path = (
                original_path
            )

            paginas_processadas = (
                total_original_pages
            )

            paginas_texto = "todas"

        else:

            paginas = sorted(
                paginas_configuradas
            )

            input_path = (
                temporary_dir
                / original_path.name
            )

            criar_pdf_selecionado(
                original_path,
                paginas,
                input_path,
            )

            paginas_processadas = (
                len(paginas)
            )

            paginas_texto = ",".join(
                str(pagina)
                for pagina in paginas
            )

        documentos.append({

            "original_path":
                original_path,

            "input_path":
                input_path,

            "arquivo":
                original_path.name,

            "paginas_originais":
                total_original_pages,

            "paginas_processadas":
                paginas_processadas,

            "paginas_selecionadas":
                paginas_texto,

            "tamanho_original_bytes":
                original_path.stat().st_size,

            "tamanho_entrada_bytes":
                input_path.stat().st_size,
        })

    return documentos


# ============================================================
# DIRETÓRIOS
# ============================================================

def preparar_diretorios():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    EXTRACTED_TEXT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# TEXTO
# ============================================================

def salvar_texto(
    library_name,
    original_path,
    text,
):

    library_dir = (
        EXTRACTED_TEXT_DIR
        / library_name
    )

    library_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        library_dir
        / (
            original_path.stem
            + OUTPUT_EXTENSION
        )
    )

    output_path.write_text(
        text,
        encoding=OUTPUT_ENCODING,
    )

    return output_path


def calcular_metricas_texto(
    text
):

    return {

        "caracteres":
            len(text),

        "palavras":
            len(text.split()),

        "linhas":
            len(
                text.splitlines()
            ),
    }


# ============================================================
# ERROS
# ============================================================

def registrar_erro(
    library_name,
    stage,
    pdf_path,
    error,
):

    file_name = (
        pdf_path.name
        if pdf_path is not None
        else ""
    )

    content = (
        "\n"
        + "=" * 80
        + "\n"
        + f"Biblioteca: {library_name}\n"
        + f"Etapa: {stage}\n"
        + f"Arquivo: {file_name}\n"
        + f"Erro: {type(error).__name__}\n"
        + f"Mensagem: {error}\n\n"
        + traceback.format_exc()
        + "\n"
    )

    with ERROR_LOG_PATH.open(
        "a",
        encoding=OUTPUT_ENCODING,
    ) as file:

        file.write(
            content
        )


# ============================================================
# RESULTADO DE ERRO
# ============================================================

def criar_resultado_erro(
    library_name,
    stage,
    error,
    documento=None,
):

    return {

        "biblioteca":
            library_name,

        "arquivo":
            (
                documento["arquivo"]
                if documento
                else ""
            ),

        "caminho_original":
            (
                str(
                    documento[
                        "original_path"
                    ]
                )
                if documento
                else ""
            ),

        "paginas_originais":
            (
                documento[
                    "paginas_originais"
                ]
                if documento
                else ""
            ),

        "paginas_processadas":
            (
                documento[
                    "paginas_processadas"
                ]
                if documento
                else ""
            ),

        "paginas_selecionadas":
            (
                documento[
                    "paginas_selecionadas"
                ]
                if documento
                else ""
            ),

        "tamanho_original_bytes":
            (
                documento[
                    "tamanho_original_bytes"
                ]
                if documento
                else ""
            ),

        "tamanho_entrada_bytes":
            (
                documento[
                    "tamanho_entrada_bytes"
                ]
                if documento
                else ""
            ),

        "repeticao":
            0,

        "status":
            "erro",

        "etapa":
            stage,

        "tempo_segundos":
            "",

        "cpu_medio_percent":
            "",

        "ram_media_mb":
            "",

        "caracteres":
            "",

        "palavras":
            "",

        "linhas":
            "",

        "arquivo_saida":
            "",

        "erro_tipo":
            type(error).__name__,

        "erro_mensagem":
            str(error),
    }


# ============================================================
# PROCESSO ISOLADO
# ============================================================

def executar_extracao_isolada(
    library_name,
    input_path,
    result_path,
    queue,
):

    extractor = None

    resource_monitor = None

    start_time = None

    try:

        gc.collect()

        extractor = (
            ExtractorFactory.create(
                library_name
            )
        )

        extractor.setup()

        resource_monitor = (
            ResourceMonitor()
        )

        resource_monitor.start()

        start_time = (
            time.perf_counter()
        )

        text = (
            extractor.run(
                str(input_path)
            )
        )

        elapsed_time = (
            time.perf_counter()
            - start_time
        )

        resource_metrics = (
            resource_monitor.stop()
        )

        # ----------------------------------------------------
        # O texto NÃO é enviado pela Queue.
        #
        # Ele é salvo diretamente em disco para evitar
        # deadlock quando o texto extraído é grande.
        # ----------------------------------------------------

        result_path.write_text(
            text,
            encoding=OUTPUT_ENCODING,
        )

        try:

            extractor.teardown()

        except Exception:

            pass

        queue.put({

            "status":
                "sucesso",

            "tempo_segundos":
                elapsed_time,

            "cpu_medio_percent":
                resource_metrics[
                    "cpu_medio_percent"
                ],

            "ram_media_mb":
                resource_metrics[
                    "ram_media_mb"
                ],

            "erro_tipo":
                "",

            "erro_mensagem":
                "",
        })

    except Exception as error:

        elapsed_time = 0.0

        if start_time is not None:

            elapsed_time = (
                time.perf_counter()
                - start_time
            )

        resource_metrics = {

            "cpu_medio_percent":
                0.0,

            "ram_media_mb":
                0.0,
        }

        if resource_monitor is not None:

            try:

                resource_metrics = (
                    resource_monitor.stop()
                )

            except Exception:

                pass

        try:

            if extractor is not None:

                extractor.teardown()

        except Exception:

            pass

        queue.put({

            "status":
                "erro",

            "tempo_segundos":
                elapsed_time,

            "cpu_medio_percent":
                resource_metrics[
                    "cpu_medio_percent"
                ],

            "ram_media_mb":
                resource_metrics[
                    "ram_media_mb"
                ],

            "erro_tipo":
                type(error).__name__,

            "erro_mensagem":
                str(error),
        })


# ============================================================
# EXECUÇÃO DE UMA REPETIÇÃO
# ============================================================

def executar_repeticao_isolada(
    library_name,
    input_path,
):

    context = mp.get_context(
        "spawn"
    )

    queue = (
        context.Queue()
    )

    result_path = (
        Path(
            tempfile.gettempdir()
        )
        / (
            "pdf_benchmark_"
            + uuid.uuid4().hex
            + ".txt"
        )
    )

    process = context.Process(
        target=executar_extracao_isolada,
        args=(
            library_name,
            str(input_path),
            result_path,
            queue,
        ),
    )

    process.start()

    # --------------------------------------------------------
    # NÃO usamos process.join() diretamente.
    #
    # O processo é acompanhado enquanto está executando,
    # permitindo aplicar timeout.
    # --------------------------------------------------------

    start_wait = time.perf_counter()

    while process.is_alive():

        elapsed_wait = (
            time.perf_counter()
            - start_wait
        )

        if (
            elapsed_wait
            >= PROCESS_TIMEOUT_SECONDS
        ):

            print(
                "\n    TIMEOUT: "
                f"{PROCESS_TIMEOUT_SECONDS} s"
            )

            process.terminate()

            process.join(
                timeout=10
            )

            if process.is_alive():

                process.kill()

                process.join()

            queue.close()

            if result_path.exists():

                result_path.unlink()

            return {

                "status":
                    "erro",

                "text":
                    "",

                "tempo_segundos":
                    PROCESS_TIMEOUT_SECONDS,

                "cpu_medio_percent":
                    "",

                "ram_media_mb":
                    "",

                "erro_tipo":
                    "TimeoutError",

                "erro_mensagem":
                    (
                        "A execução ultrapassou "
                        f"{PROCESS_TIMEOUT_SECONDS} "
                        "segundos."
                    ),
            }

        time.sleep(
            0.2
        )

    process.join(
        timeout=5
    )

    result = None

    # --------------------------------------------------------
    # Agora o processo já terminou.
    #
    # A Queue contém somente alguns números e strings curtas,
    # portanto não existe o problema de buffer causado pelo
    # texto completo.
    # --------------------------------------------------------

    try:

        if not queue.empty():

            result = (
                queue.get(
                    timeout=2
                )
            )

    except Exception:

        result = None

    queue.close()

    if result is None:

        exit_code = (
            process.exitcode
        )

        if exit_code is None:

            error_message = (
                "Processo de extração "
                "não retornou resultado."
            )

        elif exit_code < 0:

            error_message = (
                "Processo encerrado por "
                f"sinal {-exit_code}."
            )

        else:

            error_message = (
                "Processo terminou sem "
                "retornar resultado. "
                f"Exit code: {exit_code}"
            )

        if result_path.exists():

            result_path.unlink()

        return {

            "status":
                "erro",

            "text":
                "",

            "tempo_segundos":
                "",

            "cpu_medio_percent":
                "",

            "ram_media_mb":
                "",

            "erro_tipo":
                "ChildProcessError",

            "erro_mensagem":
                error_message,
        }

    # --------------------------------------------------------
    # Recupera o texto salvo pelo processo filho.
    # --------------------------------------------------------

    if result["status"] == "sucesso":

        try:

            text = (
                result_path.read_text(
                    encoding=OUTPUT_ENCODING
                )
            )

        except Exception as error:

            result = {

                "status":
                    "erro",

                "text":
                    "",

                "tempo_segundos":
                    result[
                        "tempo_segundos"
                    ],

                "cpu_medio_percent":
                    result[
                        "cpu_medio_percent"
                    ],

                "ram_media_mb":
                    result[
                        "ram_media_mb"
                    ],

                "erro_tipo":
                    type(error).__name__,

                "erro_mensagem":
                    (
                        "Falha ao ler "
                        "resultado temporário: "
                        f"{error}"
                    ),
            }

            text = ""

    else:

        text = ""

    # --------------------------------------------------------
    # Remove o arquivo temporário.
    # --------------------------------------------------------

    if result_path.exists():

        try:

            result_path.unlink()

        except Exception:

            pass

    result["text"] = text

    return result


# ============================================================
# WARM-UP
# ============================================================

def executar_warmup_isolado(
    library_name,
    input_path,
):

    print(
        "    Executando warm-up..."
    )

    result = (
        executar_repeticao_isolada(
            library_name,
            input_path,
        )
    )

    if result["status"] == "erro":

        raise RuntimeError(
            "Falha durante o warm-up: "
            f"{result['erro_tipo']}: "
            f"{result['erro_mensagem']}"
        )

    print(
        "    Warm-up concluído."
    )


# ============================================================
# CSV DETALHADO
# ============================================================

def salvar_resultados_csv(
    results
):

    if not results:

        return

    fieldnames = [

        "biblioteca",

        "arquivo",

        "caminho_original",

        "paginas_originais",

        "paginas_processadas",

        "paginas_selecionadas",

        "tamanho_original_bytes",

        "tamanho_entrada_bytes",

        "repeticao",

        "status",

        "etapa",

        "tempo_segundos",

        "cpu_medio_percent",

        "ram_media_mb",

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
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            results
        )


# ============================================================
# RESUMO
# ============================================================

def salvar_resumo_csv(
    results
):

    valid_results = [
        result
        for result in results
        if result["status"]
        in {
            "sucesso",
            "vazio",
        }
    ]

    grouped = {}

    for result in valid_results:

        grouped.setdefault(
            result["biblioteca"],
            [],
        ).append(
            result
        )

    summary = []

    for (
        library_name,
        library_results,
    ) in grouped.items():

        times = [
            result[
                "tempo_segundos"
            ]
            for result
            in library_results
            if result[
                "tempo_segundos"
            ] != ""
        ]

        chars = [
            result[
                "caracteres"
            ]
            for result
            in library_results
            if result[
                "caracteres"
            ] != ""
        ]

        cpu_values = [
            result[
                "cpu_medio_percent"
            ]
            for result
            in library_results
            if result[
                "cpu_medio_percent"
            ] != ""
        ]

        ram_values = [
            result[
                "ram_media_mb"
            ]
            for result
            in library_results
            if result[
                "ram_media_mb"
            ] != ""
        ]

        total_pages = sum(
            result[
                "paginas_processadas"
            ]
            for result
            in library_results
            if result[
                "paginas_processadas"
            ] != ""
        )

        total_time = sum(
            times
        )

        empty_extractions = sum(
            1
            for result
            in library_results
            if result["caracteres"] == 0
        )

        summary.append({

            "biblioteca":
                library_name,

            "execucoes_validas":
                len(
                    library_results
                ),

            "execucoes_vazias":
                empty_extractions,

            "tempo_total_segundos":
                total_time,

            "tempo_medio_segundos":
                (
                    statistics.mean(
                        times
                    )
                    if times
                    else 0.0
                ),

            "tempo_mediano_segundos":
                (
                    statistics.median(
                        times
                    )
                    if times
                    else 0.0
                ),

            "tempo_minimo_segundos":
                (
                    min(times)
                    if times
                    else 0.0
                ),

            "tempo_maximo_segundos":
                (
                    max(times)
                    if times
                    else 0.0
                ),

            "desvio_padrao_segundos":
                (
                    statistics.stdev(
                        times
                    )
                    if len(times) > 1
                    else 0.0
                ),

            "cpu_medio_percent":
                (
                    statistics.mean(
                        cpu_values
                    )
                    if cpu_values
                    else 0.0
                ),

            "ram_media_mb":
                (
                    statistics.mean(
                        ram_values
                    )
                    if ram_values
                    else 0.0
                ),

            "paginas_processadas":
                total_pages,

            "paginas_por_segundo":
                (
                    total_pages
                    / total_time
                    if total_time > 0
                    else 0.0
                ),

            "caracteres_medio":
                (
                    statistics.mean(
                        chars
                    )
                    if chars
                    else 0.0
                ),
        })

    fieldnames = [

        "biblioteca",

        "execucoes_validas",

        "execucoes_vazias",

        "tempo_total_segundos",

        "tempo_medio_segundos",

        "tempo_mediano_segundos",

        "tempo_minimo_segundos",

        "tempo_maximo_segundos",

        "desvio_padrao_segundos",

        "cpu_medio_percent",

        "ram_media_mb",

        "paginas_processadas",

        "paginas_por_segundo",

        "caracteres_medio",
    ]

    with SUMMARY_CSV_PATH.open(
        "w",
        newline="",
        encoding=OUTPUT_ENCODING,
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            summary
        )


# ============================================================
# EXECUÇÃO DE UMA BIBLIOTECA
# ============================================================

def executar_biblioteca(
    library_name,
    documentos,
):

    results = []

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"Biblioteca: "
        f"{library_name}"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # WARM-UP
    # --------------------------------------------------------

    try:

        if WARMUP_RUNS > 0:

            warmup_document = (
                documentos[0]
            )

            warmup_path = (
                warmup_document[
                    "input_path"
                ]
            )

            for _ in range(
                WARMUP_RUNS
            ):

                executar_warmup_isolado(
                    library_name,
                    warmup_path,
                )

    except Exception as error:

        print(
            "Erro no warm-up:"
            f" {error}"
        )

        registrar_erro(
            library_name,
            "warmup",
            warmup_document[
                "original_path"
            ],
            error,
        )

        results.append(
            criar_resultado_erro(
                library_name,
                "warmup",
                error,
                warmup_document,
            )
        )

        if FAIL_FAST:

            raise

        return results

    # --------------------------------------------------------
    # DOCUMENTOS
    # --------------------------------------------------------

    for file_index, documento in enumerate(
        documentos,
        start=1,
    ):

        original_path = (
            documento[
                "original_path"
            ]
        )

        input_path = (
            documento[
                "input_path"
            ]
        )

        print(
            f"\n[{file_index}/"
            f"{len(documentos)}] "
            f"{documento['arquivo']}"
        )

        print(
            "  Páginas processadas: "
            f"{documento['paginas_selecionadas']}"
        )

        # ----------------------------------------------------
        # REPETIÇÕES
        # ----------------------------------------------------

        for repetition in range(
            1,
            MEASURED_RUNS + 1,
        ):

            print(
                "  Repetição "
                f"{repetition}/"
                f"{MEASURED_RUNS}",
                end="",
                flush=True,
            )

            try:

                result = (
                    executar_repeticao_isolada(
                        library_name,
                        input_path,
                    )
                )

                if result["status"] == "erro":

                    print(
                        " — ERRO: "
                        f"{result['erro_mensagem']}"
                    )

                    error = RuntimeError(
                        result[
                            "erro_mensagem"
                        ]
                    )

                    registrar_erro(
                        library_name,
                        "extraction",
                        original_path,
                        error,
                    )

                    results.append({

                        "biblioteca":
                            library_name,

                        "arquivo":
                            documento[
                                "arquivo"
                            ],

                        "caminho_original":
                            str(
                                original_path
                            ),

                        "paginas_originais":
                            documento[
                                "paginas_originais"
                            ],

                        "paginas_processadas":
                            documento[
                                "paginas_processadas"
                            ],

                        "paginas_selecionadas":
                            documento[
                                "paginas_selecionadas"
                            ],

                        "tamanho_original_bytes":
                            documento[
                                "tamanho_original_bytes"
                            ],

                        "tamanho_entrada_bytes":
                            documento[
                                "tamanho_entrada_bytes"
                            ],

                        "repeticao":
                            repetition,

                        "status":
                            "erro",

                        "etapa":
                            "extraction",

                        "tempo_segundos":
                            result[
                                "tempo_segundos"
                            ],

                        "cpu_medio_percent":
                            result[
                                "cpu_medio_percent"
                            ],

                        "ram_media_mb":
                            result[
                                "ram_media_mb"
                            ],

                        "caracteres":
                            "",

                        "palavras":
                            "",

                        "linhas":
                            "",

                        "arquivo_saida":
                            "",

                        "erro_tipo":
                            result[
                                "erro_tipo"
                            ],

                        "erro_mensagem":
                            result[
                                "erro_mensagem"
                            ],
                    })

                    if FAIL_FAST:

                        raise error

                    continue

                text = result[
                    "text"
                ]

                metrics = (
                    calcular_metricas_texto(
                        text
                    )
                )

                status = (
                    "vazio"
                    if metrics[
                        "caracteres"
                    ] == 0
                    else "sucesso"
                )

                output_path = ""

                should_save = (
                    SAVE_EXTRACTED_TEXT
                    and (
                        not SAVE_ONLY_FIRST_MEASURED_RUN
                        or repetition == 1
                    )
                )

                if should_save:

                    saved_path = (
                        salvar_texto(
                            library_name,
                            original_path,
                            text,
                        )
                    )

                    output_path = str(
                        saved_path
                    )

                results.append({

                    "biblioteca":
                        library_name,

                    "arquivo":
                        documento[
                            "arquivo"
                        ],

                    "caminho_original":
                        str(
                            original_path
                        ),

                    "paginas_originais":
                        documento[
                            "paginas_originais"
                        ],

                    "paginas_processadas":
                        documento[
                            "paginas_processadas"
                        ],

                    "paginas_selecionadas":
                        documento[
                            "paginas_selecionadas"
                        ],

                    "tamanho_original_bytes":
                        documento[
                            "tamanho_original_bytes"
                        ],

                    "tamanho_entrada_bytes":
                        documento[
                            "tamanho_entrada_bytes"
                        ],

                    "repeticao":
                        repetition,

                    "status":
                        status,

                    "etapa":
                        "extraction",

                    "tempo_segundos":
                        result[
                            "tempo_segundos"
                        ],

                    "cpu_medio_percent":
                        result[
                            "cpu_medio_percent"
                        ],

                    "ram_media_mb":
                        result[
                            "ram_media_mb"
                        ],

                    "caracteres":
                        metrics[
                            "caracteres"
                        ],

                    "palavras":
                        metrics[
                            "palavras"
                        ],

                    "linhas":
                        metrics[
                            "linhas"
                        ],

                    "arquivo_saida":
                        output_path,

                    "erro_tipo":
                        "",

                    "erro_mensagem":
                        "",
                })

                print(
                    f" — "
                    f"{result['tempo_segundos']:.4f} s"
                    f" — "
                    f"{metrics['caracteres']} "
                    "caracteres"
                    f" — CPU "
                    f"{result['cpu_medio_percent']:.2f}%"
                    f" — RAM "
                    f"{result['ram_media_mb']:.2f} MB"
                )

            except Exception as error:

                print(
                    " — ERRO: "
                    f"{error}"
                )

                registrar_erro(
                    library_name,
                    "extraction",
                    original_path,
                    error,
                )

                results.append(
                    criar_resultado_erro(
                        library_name,
                        "extraction",
                        error,
                        documento,
                    )
                )

                if FAIL_FAST:

                    raise

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    mp.freeze_support()

    preparar_diretorios()

    ERROR_LOG_PATH.write_text(
        "",
        encoding=OUTPUT_ENCODING,
    )

    pdf_files = (
        listar_pdfs()
    )

    print(
        "\nBENCHMARK DE "
        "EXTRAÇÃO TEXTUAL DE PDFs"
    )

    print(
        "=" * 70
    )

    print(
        f"Dataset: "
        f"{DATASET_DIR}"
    )

    print(
        f"PDFs encontrados: "
        f"{len(pdf_files)}"
    )

    print(
        f"Bibliotecas: "
        f"{len(ENABLED_EXTRACTORS)}"
    )

    print(
        f"Warm-ups: "
        f"{WARMUP_RUNS}"
    )

    print(
        f"Repetições medidas: "
        f"{MEASURED_RUNS}"
    )

    print(
        "Processo isolado: "
        "1 processo por repetição"
    )

    print(
        "Timeout por execução: "
        f"{PROCESS_TIMEOUT_SECONDS} s"
    )

    print(
        "Monitoramento: "
        "CPU média + RAM média"
    )

    print(
        "Saída: TXT UTF-8"
    )

    with tempfile.TemporaryDirectory(
        prefix="pdf_benchmark_pages_"
    ) as temp_dir:

        temporary_dir = Path(
            temp_dir
        )

        documentos = (
            preparar_documentos(
                pdf_files,
                temporary_dir,
            )
        )

        all_results = []

        for library_name in (
            ENABLED_EXTRACTORS
        ):

            library_results = (
                executar_biblioteca(
                    library_name,
                    documentos,
                )
            )

            all_results.extend(
                library_results
            )

            salvar_resultados_csv(
                all_results
            )

            salvar_resumo_csv(
                all_results
            )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Benchmark concluído."
    )

    print(
        "Resultados detalhados: "
        f"{RESULTS_CSV_PATH}"
    )

    print(
        "Resumo: "
        f"{SUMMARY_CSV_PATH}"
    )

    print(
        "Textos extraídos: "
        f"{EXTRACTED_TEXT_DIR}"
    )

    print(
        "Log de erros: "
        f"{ERROR_LOG_PATH}"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()