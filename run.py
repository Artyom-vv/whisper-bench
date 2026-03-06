"""
Таск-раннер для faster-whisper benchmark.
Использование:
    python run.py                  # показать все задачи с расчётным временем под текущий GPU
    python run.py smoke            # быстрая проверка скрипта
    python run.py light            # тест модели small
    python run.py medium           # тест модели medium
    python run.py heavy            # тест модели large-v3
    python run.py full             # полный прогон всех моделей
    python run.py batch            # поиск оптимального batch_size
    python run.py vad              # измерение стоимости VAD
    python run.py stress           # стресс-тест на длинных файлах

Флаги:
    --dry-run      показать команду без запуска
    --skip-errors  продолжать при OOM/ошибках
    --output DIR   переопределить папку результатов
"""

import argparse
import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

try:
    import yaml as yaml
    _YAML = True
except ImportError:
    import types
    yaml = types.ModuleType("yaml")  # заглушка — реальный вызов защищён _YAML
    _YAML = False

BENCHMARKER = "benchmarker_batch.py"

# ─── Реестр задач ────────────────────────────────────────────────────────────
# (configs, manifest, output_subdir, описание)

TASKS = {
    "smoke": (
        "configs/configs_smoke.yaml",
        "manifests/manifest_smoke.yaml",
        "results/smoke",
        "Проверка скрипта: 1 EN + 1 RU, 2 мин",
        True
    ),
    "light": (
        "configs/configs_light.yaml",
        "manifests/manifest_short.yaml",
        "results/light",
        "Модель small: int8/float16/int8_float16, beam 1 и 5",
        False
    ),
    "medium": (
        "configs/configs_medium.yaml",
        "manifests/manifest_short.yaml",
        "results/medium",
        "Модель medium: int8/float16/int8_float16, beam 1 и 5",
        False
    ),
    "heavy": (
        "configs/configs_heavy.yaml",
        "manifests/manifest_short.yaml",
        "results/heavy",
        "Модель large-v3: int8/int8_float16, beam 1 и 5",
        False
    ),
    "full": (
        "configs/configs_full.yaml",
        "manifests/manifest_mid.yaml",
        "results/full",
        "Все модели, batch_size=8, manifest_mid (5+15 мин аудио)",
        False
    ),
    "batch": (
        "configs/configs_batch.yaml",
        "manifests/manifest_mid.yaml",
        "results/batch",
        "Поиск оптимального batch_size на large-v3 int8",
        False
    ),
    "vad": (
        "configs/configs_vad_overhead.yaml",
        "manifests/manifest_mid.yaml",
        "results/vad_overhead",
        "Измерение стоимости VAD (справочно, запускать один раз)",
        False
    ),
    "stress": (
        "configs/configs_heavy.yaml",
        "manifests/manifest_long.yaml",
        "results/stress",
        "Стресс-тест large-v3 на 30-мин и 60-мин файлах",
        False
    ),
}

# ─── GPU detection ───────────────────────────────────────────────────────────

# Таблица известных GPU: подстрока в названии → (bandwidth GB/s, SM count)
# Bandwidth определяет скорость чтения весов — ключевой фактор для LLM-инференса.
_GPU_TABLE = {
    "H100":          (3350, 132),
    "A100 80":       (2000, 108),
    "A100 40":       (1555, 108),
    "A40":           (696,  84),
    "A10":           (600,  72),
    "V100":          (900,  80),
    "T4":            (300,  40),
    "4090":          (1008, 128),
    "4080 SUPER":    (736,  80),
    "4080":          (717,  76),
    "4070 Ti SUPER": (672,  66),
    "4070 Ti":       (504,  60),
    "4070 SUPER":    (504,  56),
    "4070":          (504,  46),
    "4060 Ti 16":    (288,  34),
    "4060 Ti":       (288,  34),
    "4060":          (272,  24),
    "3090 Ti":       (1008, 84),
    "3090":          (936,  82),
    "3080 Ti":       (912,  80),
    "3080 12":       (912,  70),
    "3080":          (760,  68),
    "3070 Ti":       (608,  48),
    "3070":          (448,  46),
    "3060 Ti":       (448,  38),
    "3060":          (360,  28),
    "3050":          (224,  20),
    "2080 Ti":       (616,  68),
    "2080 SUPER":    (496,  48),
    "2080":          (448,  46),
    "2070 SUPER":    (448,  40),
    "2070":          (448,  36),
    "2060 SUPER":    (448,  34),
    "2060":          (336,  30),
    "1080 Ti":       (484,  28),
    "1080":          (320,  20),
    "1070 Ti":       (256,  19),
    "1070":          (256,  15),
}

# Базовый RTF large-v3 int8 bs=1 beam=5 на RTX 3050 (224 GB/s, 20 SM) — из наших тестов
_REFERENCE_RTF   = 0.083
_REFERENCE_BW    = 224.0   # GB/s
_REFERENCE_SM    = 20

# Масштабирующие коэффициенты относительно large-v3 int8 bs=1 beam=5
# Получены из эмпирики + архитектурных пропорций параметров
_MODEL_SCALE = {
    "large-v3":  1.00,
    "large-v2":  1.00,
    "medium":    0.38,
    "small":     0.17,
    "base":      0.08,
    "tiny":      0.04,
}
_COMPUTE_SCALE = {
    "int8":         1.00,
    "int8_float16": 0.90,
    "float16":      1.15,
    "float32":      2.20,
}
_BEAM_SCALE = {1: 0.58, 2: 0.75, 3: 0.87, 4: 0.94, 5: 1.00}

# batch_size scaling: при bs>1 RTF падает до плато насыщения GPU
def _batch_scale(bs: int, sm: int) -> float:
    sat = max(1, round(sm / 2.5))   # bs насыщения по числу SM
    if bs <= 1:
        return 1.00
    eff = min(bs, sat) / sat        # доля использования GPU
    return 1.0 - 0.72 * eff         # эмпирический коэффициент из наших тестов

def _batch_candidates(gpu: dict) -> list[int]:
    sat = max(1, round(gpu["sm"] / 2.5))
    vram_max_bs = max(1, (gpu["vram_mb"] - 2000) // 400)
    raw = sorted({1, sat // 2, sat, sat + 4, sat * 2})
    return [bs for bs in raw if 1 <= bs <= vram_max_bs]


def _write_batch_config(gpu: dict, path: str) -> list[int]:
    """Генерирует configs_batch_dynamic.yaml под текущий GPU. Возвращает список bs."""
    candidates = _batch_candidates(gpu)
    configs = []
    for bs in candidates:
        configs.append({
            "id":            f"lv3_int8_bs{bs}",
            "model":         "large-v3",
            "compute_type":  "int8",
            "vad_filter":    True,
            "language_mode": "auto",
            "beam_size":     5,
            "batch_size":    bs,
            "chunk_length":  30,
        })
    data = {
        "device":                    "cuda",
        "runs_per_case":             1,
        "gpu_price_per_hour":        0.0,
        "config_timeout_seconds":    600,
        "per_audio_timeout_seconds": 300,
        "configurations":            configs,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)
    return candidates


def _detect_gpu() -> dict:
    """Возвращает dict с именем GPU, bandwidth и SM count."""
    try:
        si = None
        if os.name == "nt":
            import subprocess as _sp
            si = _sp.STARTUPINFO()
            si.dwFlags |= _sp.STARTF_USESHOWWINDOW
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            text=True, timeout=5, startupinfo=si,
        ).strip().split("\n")[0]
        name, vram_str = out.split(",", 1)
        name    = name.strip()
        vram_mb = int(vram_str.strip().replace(" MiB", "").replace(" MB", ""))
    except Exception:
        return {"name": "Unknown GPU", "bw": _REFERENCE_BW,
                "sm": _REFERENCE_SM, "vram_mb": 0}

    bw, sm = _REFERENCE_BW, _REFERENCE_SM
    for key, (k_bw, k_sm) in _GPU_TABLE.items():
        if key.lower() in name.lower():
            bw, sm = k_bw, k_sm
            break

    return {"name": name, "bw": bw, "sm": sm, "vram_mb": vram_mb}


# ─── ETA estimation ──────────────────────────────────────────────────────────

def _parse_task_params(configs_path: str, manifest_path: str):
    """Читает YAML и возвращает (список конфигов, суммарную длительность аудио)."""
    if not _YAML:
        return [], 1, 0
    try:
        with open(configs_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        with open(manifest_path, encoding="utf-8") as f:
            mfst = yaml.safe_load(f)
    except Exception:
        return [], 1, 0

    configs   = cfg.get("configurations", [])
    runs      = cfg.get("runs_per_case", 1)
    audio_sec = sum(a.get("duration_seconds", 0)
                    for a in mfst.get("audio_files", []))
    return configs, runs, audio_sec


def _estimate_eta(configs_path: str, manifest_path: str, gpu: dict) -> str:
    """Возвращает строку с расчётным временем для задачи под текущий GPU."""
    configs, runs, audio_sec = _parse_task_params(configs_path, manifest_path)
    if not configs or not audio_sec:
        return "?"

    bw_scale = _REFERENCE_BW / max(gpu["bw"], 1)
    sm       = gpu["sm"]

    total_s = 0.0
    for c in configs:
        model   = c.get("model", "large-v3")
        ctype   = c.get("compute_type", "int8")
        bs      = c.get("batch_size", 1)
        beam    = c.get("beam_size", 5)

        rtf = (
            _REFERENCE_RTF
            * _MODEL_SCALE.get(model, 1.0)
            * _COMPUTE_SCALE.get(ctype, 1.0)
            * _BEAM_SCALE.get(min(beam, 5), 1.0)
            * _batch_scale(bs, sm)
            * bw_scale
        )
        # +30 сек overhead на загрузку модели + выгрузку
        total_s += audio_sec * rtf * runs + 30

    if total_s < 120:
        return f"~{int(total_s // 60) or 1} мин"
    elif total_s < 3600:
        lo = max(1, int(total_s * 0.8 // 60))
        hi = int(total_s * 1.3 // 60) + 1
        return f"~{lo}–{hi} мин"
    else:
        lo = round(total_s * 0.8 / 3600, 1)
        hi = round(total_s * 1.3 / 3600, 1)
        return f"~{lo}–{hi} ч"

def _estimate_eta_batch(gpu: dict, manifest_path: str) -> str:
    candidates = _batch_candidates(gpu)
    if not _YAML:
        return "?"
    try:
        with open(manifest_path, encoding="utf-8") as f:
            mfst = yaml.safe_load(f)
    except Exception:
        return "?"
    audio_sec = sum(a.get("duration_seconds", 0) for a in mfst.get("audio_files", []))
    bw_scale  = _REFERENCE_BW / max(gpu["bw"], 1)
    total_s   = 0.0
    for bs in candidates:
        rtf = (
            _REFERENCE_RTF
            * _COMPUTE_SCALE["int8"]
            * _BEAM_SCALE[5]
            * _batch_scale(bs, gpu["sm"])
            * bw_scale
        )
        total_s += audio_sec * rtf + 30
    lo = max(1, int(total_s * 0.8 // 60))
    hi = int(total_s * 1.3 // 60) + 1
    return f"~{lo}–{hi} мин"

# ─── Helpers ─────────────────────────────────────────────────────────────────

def print_tasks(gpu: dict):
    etas = {}
    for name, (cfg, mfst, _, _, _) in TASKS.items():
        etas[name] = (
            _estimate_eta_batch(gpu, TASKS["batch"][1])
            if name == "batch"
            else _estimate_eta(cfg, mfst, gpu)
        )

    w_name = max(len(n)    for n in TASKS)              + 2
    w_desc = max(len(v[3]) for v in TASKS.values())     + 2
    w_eta  = max(len(e)    for e in etas.values())      + 2
    total  = w_name + w_desc + w_eta + 6

    gpu_line = f"{gpu['name']}  |  {gpu['bw']} GB/s bandwidth  |  {gpu['sm']} SM  |  {gpu['vram_mb']} MiB VRAM"

    print()
    print(f"  GPU: {gpu_line}")
    print(f"  {'Задача':<{w_name}}  {'Описание':<{w_desc}}  {'ETA':<{w_eta}}")
    print("  " + "─" * (total - 2))
    for name, (_, _, _, desc, _) in TASKS.items():
        print(
            f"  {name:<{w_name}}"
            f"  {desc:<{w_desc}}"
            f"  {etas[name]:<{w_eta}}"
        )
    print()


def run_task(name: str, output_override: str | None,
             dry_run: bool, skip_errors: bool, gpu: dict):
    if name not in TASKS:
        print(f"[ERROR] Неизвестная задача: \"{name}\"")
        print_tasks(gpu)
        sys.exit(1)

    configs, manifest, out_dir, desc, single_mode = TASKS[name]
    eta = _estimate_eta(configs, manifest, gpu)

    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = output_override or f"{out_dir}_{ts}"

    DYNAMIC_BATCH_CFG = "configs/configs_batch_dynamic.yaml"

    if name == "batch":
        candidates = _write_batch_config(gpu, DYNAMIC_BATCH_CFG)
        print(f"[BATCH] GPU: {gpu['name']} → batch_size candidates: {candidates}")
        print(f"[BATCH] конфиг записан → {DYNAMIC_BATCH_CFG}")
        configs = DYNAMIC_BATCH_CFG   # переопределяем локальную переменную

    cmd = [
        sys.executable, BENCHMARKER,
        "--manifest",   manifest,
        "--configs",    configs,
        "--output-dir", output,
    ]
    if single_mode:
        cmd.append("--single-mode")
    if skip_errors:
        cmd.append("--skip-on-error")

    print(f"\n[TASK]  {name}: {desc}")
    print(f"[GPU]   {gpu['name']} ({gpu['bw']} GB/s, {gpu['sm']} SM)")
    print(f"[ETA]   {eta}")
    print(f"[OUT]   {output}")
    print(f"[CMD]   {' '.join(cmd)}\n")

    if dry_run:
        print("[DRY-RUN] Команда не выполнена.")
        return

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="faster-whisper benchmark task runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Доступные задачи: " + ", ".join(TASKS),
    )
    parser.add_argument("task",          nargs="?",        help="Название задачи")
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument("--output",      default=None)
    args = parser.parse_args()

    gpu = _detect_gpu()

    if not args.task:
        print_tasks(gpu)
        return

    run_task(args.task, args.output, args.dry_run, args.skip_errors, gpu)


if __name__ == "__main__":
    main()
