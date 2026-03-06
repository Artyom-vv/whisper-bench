import argparse
import csv
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

import torch
import yaml
from faster_whisper import BatchedInferencePipeline, WhisperModel

try:
    import psutil as _psutil
except ImportError:
    _psutil = None  # type: ignore[assignment]


# ─── Трекер памяти (VRAM + RAM) ──────────────────────────────────────────────

class MemoryTracker:
    """
    Отслеживает пиковую VRAM и RAM в фоновом потоке.

    ВАЖНО: faster-whisper использует CTranslate2, который выделяет CUDA-память
    напрямую, минуя torch-аллокатор. Поэтому torch.cuda.memory_reserved()
    всегда возвращает 0. Единственный надёжный способ — опрашивать nvidia-smi.
    """

    def __init__(self, device: str, interval: float = 0.5):
        self.device = device
        self.interval = interval
        self.peak_vram_mb: float = 0
        self.peak_ram_mb:  float = 0
        self.running = False
        self.thread: threading.Thread | None = None
        self._proc = _psutil.Process() if _psutil is not None else None

        # nvidia-smi для VRAM
        self._nvsmi_cmd: list[str] | None = None
        self._startupinfo = None
        if device == "cuda":
            self._nvsmi_cmd = [
                "nvidia-smi", "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ]
            if os.name == "nt":
                si = subprocess.STARTUPINFO()
                si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                self._startupinfo = si

    def _track(self):
        while self.running:
            if self._nvsmi_cmd:
                try:
                    out = subprocess.check_output(
                        self._nvsmi_cmd, text=True,
                        startupinfo=self._startupinfo,
                        timeout=2,
                    ).strip()
                    if out.isdigit():
                        vram = int(out)
                        if vram > self.peak_vram_mb:
                            self.peak_vram_mb = vram
                except Exception:
                    pass

            if self._proc:
                try:
                    ram = self._proc.memory_info().rss / 1024 ** 2
                    if ram > self.peak_ram_mb:
                        self.peak_ram_mb = ram
                except Exception:
                    pass

            time.sleep(self.interval)

    def __enter__(self):
        self.running = True
        self.thread = threading.Thread(target=self._track, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)


# ─── Вспомогательные функции ─────────────────────────────────────────────────

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _unload(base_model, batched_model, device: str) -> None:
    try:
        if batched_model is not None:
            del batched_model
        if base_model is not None:
            del base_model
    except Exception:
        pass
    gc.collect()
    # torch.cuda.empty_cache() — намеренно убрано:
    # CTranslate2 управляет CUDA-памятью независимо от torch.
    # empty_cache() после del model трогает уже освобождённый
    # CTranslate2-контекст → STATUS_STACK_BUFFER_OVERRUN (0xC0000409)
    time.sleep(0.5)


def _cast_row(row: dict) -> dict:
    for col in ("run_index", "batch_size", "segments_count"):
        if row.get(col):
            row[col] = int(row[col])
    for col in ("execution_time_s", "rtf", "language_probability",
                "peak_vram_mb", "peak_ram_mb", "cost_per_audio_minute"):
        if row.get(col):
            row[col] = float(row[col])
    if not row.get("error"):
        row["error"] = None
    return row


def _timeout_rows(config: dict, audio_files: list) -> list[dict]:
    return [
        {
            "config_id": config["id"],
            "audio_id": a["id"],
            "run_index": 1,
            "batch_size": config.get("batch_size", 1),
            "execution_time_s": None, "rtf": None, "segments_count": None,
            "detected_language": None, "language_probability": None,
            "peak_vram_mb": None, "peak_ram_mb": None,
            "cost_per_audio_minute": None, "error": "TIMEOUT",
        }
        for a in audio_files
    ]


def build_summary(rows: list[dict]) -> dict:
    from statistics import median

    grouped: dict = {}
    for row in rows:
        if row["error"]:
            continue
        key = f"{row['config_id']}__{row['audio_id']}"
        grouped.setdefault(key, []).append(row)

    summary: dict = {}
    for key, group in grouped.items():
        rtfs       = [r["rtf"]            for r in group]
        exec_times = [r["execution_time_s"] for r in group]
        vram_vals  = [r["peak_vram_mb"]   for r in group if r.get("peak_vram_mb")]
        ram_vals   = [r["peak_ram_mb"]    for r in group if r.get("peak_ram_mb")]
        sorted_rtfs = sorted(rtfs)
        p95_idx = max(0, int(len(sorted_rtfs) * 0.95) - 1)

        all_for_key  = [r for r in rows if f"{r['config_id']}__{r['audio_id']}" == key]
        errors_total = sum(1 for r in all_for_key if r["error"])

        summary[key] = {
            "rtf_p50":                   median(rtfs),
            "rtf_p95":                   sorted_rtfs[p95_idx],
            "exec_time_p50_s":           median(exec_times),
            "peak_vram_mb_max":          max(vram_vals) if vram_vals else None,
            "peak_ram_mb_max":           max(ram_vals)  if ram_vals  else None,
            "cost_per_audio_minute_median": median(
                [r["cost_per_audio_minute"] for r in group]
            ),
            "error_rate": errors_total / len(all_for_key) if all_for_key else 0,
        }
    return summary


FIELDNAMES = [
    "config_id", "audio_id", "run_index", "batch_size",
    "execution_time_s", "rtf", "segments_count",
    "detected_language", "language_probability",
    "peak_vram_mb", "peak_ram_mb", "cost_per_audio_minute", "error",
]


# ─── Логика одного конфига (single-mode) ─────────────────────────────────────

def run_config_directly(
    config: dict,
    audio_files: list,
    device: str,
    runs_per_case: int,
    gpu_price_per_hour: float,
    skip_on_error: bool,
    per_audio_timeout: int,
    writer=None,
    csv_file=None,
) -> list[dict]:

    config_id    = config["id"]
    model_size   = config["model"]
    compute_type = config["compute_type"]
    vad_filter   = config.get("vad_filter", False)
    language_mode = config.get("language_mode", "auto")
    beam_size    = config.get("beam_size", 5)
    batch_size   = config.get("batch_size", 1)
    chunk_length = config.get("chunk_length", 30)
    use_batched  = batch_size > 1

    if use_batched and not vad_filter:
        print(f"  [WARN] batch_size={batch_size} без vad_filter=True: "
              f"все mel-чанки файла будут в RAM одновременно!")

    print(f"\n[CONFIG] {config_id}")
    print(f"  model={model_size}, compute_type={compute_type}, "
          f"vad={vad_filter}, lang={language_mode}, "
          f"batch_size={batch_size}, chunk_length={chunk_length}s")

    base_model = None
    batched_model = None
    rows: list[dict] = []

    try:
        base_model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print(f"  [INFO] base_model loaded")

        if use_batched:
            batched_model = BatchedInferencePipeline(model=base_model)
            print(f"  [INFO] BatchedInferencePipeline ready")

    except Exception:
        print(f"  [ERROR] {traceback.format_exc()}")
        _unload(base_model, batched_model, device)
        if not skip_on_error:
            raise
        return rows

    try:
        for audio in audio_files:
            audio_id   = audio["id"]
            audio_path = audio["path"]
            duration   = audio["duration_seconds"]
            language   = None if language_mode == "auto" else audio.get("language")

            run_timeout = max(per_audio_timeout, int(duration * 15))

            print(f"\n  [AUDIO] {audio_id} | {duration}s | lang={language or 'auto'} "
                  f"| run_timeout={run_timeout}s")

            for run_idx in range(1, runs_per_case + 1):
                print(f"    run {run_idx}/{runs_per_case} ...", end=" ", flush=True)

                row: dict = {k: None for k in FIELDNAMES}
                row.update({"config_id": config_id, "audio_id": audio_id,
                            "run_index": run_idx, "batch_size": batch_size})

                result_holder: dict = {}
                exc_holder:    dict = {}
                done_event = threading.Event()

                # Захватываем значения локально, чтобы closure не тянула ref на loop-var
                _base    = base_model
                _batched = batched_model
                _device  = device
                _path    = audio_path
                _lang    = language
                _bs      = batch_size
                _bsz     = beam_size
                _vad     = vad_filter
                _chunk   = chunk_length

                def _run_transcription(
                    base=_base, batched=_batched, dev=_device,
                    path=_path, lang=_lang, bs=_bs,
                    bsz=_bsz, vad=_vad, chunk=_chunk,
                ):
                    try:
                        if dev == "cuda":
                            torch.cuda.synchronize()

                        with MemoryTracker(dev) as tracker:
                            t0 = time.perf_counter()

                            if batched is not None:
                                segs, info = batched.transcribe(
                                    path,
                                    language=lang,
                                    beam_size=bsz,
                                    batch_size=bs,
                                    vad_filter=vad,
                                    chunk_length=chunk,
                                )
                            else:
                                segs, info = base.transcribe(
                                    path,
                                    language=lang,
                                    beam_size=bsz,
                                    vad_filter=vad,
                                )

                            segs_list = list(segs)

                            if dev == "cuda":
                                torch.cuda.synchronize()
                            t1 = time.perf_counter()

                        n_segs = len(segs_list)
                        del segs_list
                        gc.collect()

                        result_holder["n_segs"] = n_segs
                        result_holder["info"]   = info
                        result_holder["exec"]   = t1 - t0
                        result_holder["vram"]   = tracker.peak_vram_mb if tracker.peak_vram_mb > 0 else None
                        result_holder["ram"]    = tracker.peak_ram_mb  if tracker.peak_ram_mb  > 0 else None

                    except torch.cuda.OutOfMemoryError:
                        exc_holder["error"] = "OOM"
                        torch.cuda.empty_cache()
                    except Exception:
                        exc_holder["error"] = traceback.format_exc(limit=2)
                    finally:
                        done_event.set()

                t = threading.Thread(target=_run_transcription, daemon=True)
                t.start()
                finished = done_event.wait(timeout=run_timeout)

                if not finished:
                    row["error"] = f"RUN_TIMEOUT>{run_timeout}s"
                    print(f"RUN_TIMEOUT ({run_timeout}s)")
                elif exc_holder.get("error"):
                    row["error"] = exc_holder["error"]
                    print(exc_holder["error"].split("\n")[0])
                    if not skip_on_error and exc_holder["error"] == "OOM":
                        raise torch.cuda.OutOfMemoryError
                else:
                    exec_time = result_holder["exec"]
                    rtf       = exec_time / duration
                    cost      = (exec_time / 3600) * gpu_price_per_hour * 60
                    info      = result_holder["info"]
                    vram_val  = result_holder.get("vram")
                    ram_val   = result_holder.get("ram")

                    row.update({
                        "execution_time_s":      round(exec_time, 3),
                        "rtf":                   round(rtf, 4),
                        "segments_count":        result_holder.get("n_segs", 0),
                        "detected_language":     info.language,
                        "language_probability":  round(info.language_probability, 4),
                        "peak_vram_mb":          vram_val,
                        "peak_ram_mb":           ram_val,
                        "cost_per_audio_minute": round(cost, 6),
                    })

                    vram_str = f"{vram_val:.0f}MB" if vram_val is not None else "n/a"
                    ram_str  = f"{ram_val:.0f}MB"  if ram_val  is not None else "n/a"
                    print(f"ok | rtf={row['rtf']} | {exec_time:.1f}s | "
                          f"VRAM={vram_str} | RAM={ram_str}")
                    
                rows.append(row)
                gc.collect()
                # if device == "cuda":
                #     torch.cuda.empty_cache()

    finally:
        if writer is not None:
            for row in rows:
                writer.writerow(row)
            if csv_file is not None:
                csv_file.flush()
                os.fsync(csv_file.fileno())

        _unload(base_model, batched_model, device)
        print(f"\n  [CONFIG DONE] {config_id} — модель выгружена")

    return rows


# ─── Оркестратор ─────────────────────────────────────────────────────────────

def run_benchmark(
    manifest_path: str,
    configs_path: str,
    output_dir: str,
    skip_on_error: bool,
    single_mode: bool = False,
    per_audio_timeout: int = 300,
):
    manifest     = load_yaml(manifest_path)
    configs_data = load_yaml(configs_path)

    audio_files    = manifest["audio_files"]
    configurations = configs_data["configurations"]
    device         = configs_data.get("device", "cuda")
    runs_per_case  = configs_data.get("runs_per_case", 1)
    gpu_price      = configs_data.get("gpu_price_per_hour", 0.0)
    config_timeout = configs_data.get("config_timeout_seconds", 1800)
    per_audio_timeout = configs_data.get(
        "per_audio_timeout_seconds", per_audio_timeout
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "results_batch.csv"

    all_rows: list[dict] = []

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()

        if single_mode:
            for config in configurations:
                rows = run_config_directly(
                    config, audio_files, device,
                    runs_per_case, gpu_price, skip_on_error, per_audio_timeout,
                    writer=writer, csv_file=csv_file
                )
                csv_file.flush()
                all_rows.extend(rows)

        else:
            for config in configurations:
                config_id = config["id"]
                print(f"\n[ORCHESTRATOR] {config_id} (config_timeout={config_timeout}s)")

                single_cfg = {
                    "device": device,
                    "runs_per_case": runs_per_case,
                    "gpu_price_per_hour": gpu_price,
                    "per_audio_timeout_seconds": per_audio_timeout,
                    "configurations": [config],
                }
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".yaml", delete=False, encoding="utf-8"
                ) as tf:
                    yaml.dump(single_cfg, tf, allow_unicode=True)
                    tmp_cfg_path = tf.name

                tmp_out = output_path / f"_tmp_{config_id}"
                cmd = [
                    sys.executable, __file__,
                    "--manifest", manifest_path,
                    "--configs", tmp_cfg_path,
                    "--output-dir", str(tmp_out),
                    "--single-mode",
                ]
                if skip_on_error:
                    cmd.append("--skip-on-error")

                proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
                timed_out = False
                try:
                    proc.wait(timeout=config_timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    proc.kill()
                    proc.wait()
                    print(f"  [TIMEOUT] {config_id} убит после {config_timeout}s")

                if timed_out:
                    for row in _timeout_rows(config, audio_files):
                        writer.writerow(row)
                        all_rows.append(row)
                else:
                    tmp_csv = tmp_out / "results_batch.csv"
                    if tmp_csv.exists():
                        with open(tmp_csv, encoding="utf-8") as f:
                            for row in csv.DictReader(f):
                                row = _cast_row(row)
                                writer.writerow(row)
                                all_rows.append(row)
                    else:
                        print(f"  [WARN] tmp CSV не найден после завершения: {tmp_csv}")
                        print(f"  [WARN] Результаты конфига {config_id} потеряны — "
                              f"subprocess вышел с кодом {proc.returncode}")

                csv_file.flush()
                shutil.rmtree(tmp_out, ignore_errors=True)
                try:
                    os.unlink(tmp_cfg_path)
                except Exception:
                    pass

    summary = build_summary(all_rows)
    summary_path = output_path / "summary_batch.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] results_batch.csv → {csv_path}")
    print(f"[DONE] summary_batch.json → {summary_path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="faster-whisper BATCH benchmark v2")
    parser.add_argument("--manifest",  required=True)
    parser.add_argument("--configs",   required=True)
    parser.add_argument("--output-dir", default="./results_batch")
    parser.add_argument("--skip-on-error", action="store_true")
    parser.add_argument("--single-mode",   action="store_true",
                        help="Внутренний режим: запустить конфиги напрямую")
    parser.add_argument("--per-audio-timeout", type=int, default=300,
                        help="Макс. секунд на один прогон (override yaml)")
    args = parser.parse_args()

    run_benchmark(
        manifest_path=args.manifest,
        configs_path=args.configs,
        output_dir=args.output_dir,
        skip_on_error=args.skip_on_error,
        single_mode=args.single_mode,
        per_audio_timeout=args.per_audio_timeout,
    )


if __name__ == "__main__":
    main()
