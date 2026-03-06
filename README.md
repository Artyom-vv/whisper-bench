# whisper-bench

Набор бенчмарков для оценки производительности `faster-whisper`: скорость (RTF), использование VRAM и RAM для разных моделей, режимов вычислений и batch size

## Установка

```bash
pip install -r requirements.txt
```

## Быстрый старт

```bash
python run.py smoke
```

## Общий список тестов

```bash
python run.py
```

Этот запуск проверяет, что окружение настроено корректно и модели успешно загружаются.
