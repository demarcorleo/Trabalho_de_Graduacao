import os
import threading
import time
from typing import Dict, List

import psutil


class ResourceMonitor:
    """
    Monitora CPU e memória do processo principal e de seus
    processos filhos durante uma execução.

    CPU:
        média percentual observada durante a execução.

    RAM:
        média de memória RSS observada durante a execução,
        em MB.

    O monitoramento inclui processos filhos porque algumas
    bibliotecas de extração podem utilizar subprocessos.
    """

    def __init__(
        self,
        interval: float = 0.05,
    ):
        self.interval = interval

        self.process = psutil.Process(
            os.getpid()
        )

        self.running = False

        self.cpu_samples: List[float] = []

        self.ram_samples: List[float] = []

        self.monitor_thread = None

    def _get_processes(self):
        """
        Retorna o processo principal e os processos filhos
        existentes naquele momento.
        """

        processes = []

        try:

            if self.process.is_running():

                processes.append(
                    self.process
                )

        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
        ):

            pass

        try:

            children = (
                self.process.children(
                    recursive=True
                )
            )

            processes.extend(
                children
            )

        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
        ):

            pass

        return processes

    def _read_resources(self):
        """
        Coleta CPU e RAM do processo principal e de seus
        processos filhos.

        Retorna:
            cpu_total: percentual total de CPU observado.
            ram_mb: memória RSS total em MB.
        """

        processes = self._get_processes()

        cpu_total = 0.0

        ram_total = 0

        for process in processes:

            try:

                cpu_total += (
                    process.cpu_percent(
                        interval=None
                    )
                )

                ram_total += (
                    process.memory_info().rss
                )

            except (
                psutil.NoSuchProcess,
                psutil.AccessDenied,
            ):

                continue

        ram_mb = (
            ram_total
            / (
                1024 * 1024
            )
        )

        return (
            cpu_total,
            ram_mb,
        )

    def _monitor_loop(self):
        """
        Loop de amostragem executado em uma thread separada.
        """

        while self.running:

            cpu, ram = (
                self._read_resources()
            )

            self.cpu_samples.append(
                cpu
            )

            self.ram_samples.append(
                ram
            )

            time.sleep(
                self.interval
            )

    def start(self):
        """
        Inicia o monitoramento.
        """

        if self.running:

            return

        self.running = True

        self.cpu_samples = []

        self.ram_samples = []

        # Inicializa as métricas de CPU do psutil.
        self._read_resources()

        self.monitor_thread = (
            threading.Thread(
                target=self._monitor_loop,
                daemon=True,
            )
        )

        self.monitor_thread.start()

    def stop(
        self,
    ) -> Dict[str, float]:
        """
        Para o monitoramento e retorna as médias.
        """

        if not self.running:

            return {
                "cpu_medio_percent": 0.0,
                "ram_media_mb": 0.0,
            }

        self.running = False

        if self.monitor_thread is not None:

            self.monitor_thread.join(
                timeout=2.0
            )

        cpu_mean = 0.0

        ram_mean = 0.0

        if self.cpu_samples:

            cpu_mean = (
                sum(
                    self.cpu_samples
                )
                / len(
                    self.cpu_samples
                )
            )

        if self.ram_samples:

            ram_mean = (
                sum(
                    self.ram_samples
                )
                / len(
                    self.ram_samples
                )
            )

        self.monitor_thread = None

        return {
            "cpu_medio_percent": cpu_mean,
            "ram_media_mb": ram_mean,
        }

    def __enter__(self):

        self.start()

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):

        self.stop()