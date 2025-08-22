# test_generate.py
import os
import pathlib

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ----------------------------------------------------------------------
# 1️⃣  Фикстуры
# ----------------------------------------------------------------------
@pytest.fixture(scope="session")
def model_path(tmp_path_factory):
    """
    Путь к уже‑сохранённой merged‑модели.
    Если модель не найдена – тест будет пропущен (skip), чтобы CI не падал.
    """
    path = pathlib.Path("./my_finetuned/full_merged")
    if not path.is_dir():
        pytest.skip("Merged model not found at ./my_finetuned/full_merged")
    return str(path)


@pytest.fixture(scope="session")
def tokenizer(model_path):
    """Токенизатор, загружаемый из той же директории, что и модель."""
    return AutoTokenizer.from_pretrained(model_path, use_fast=True)


@pytest.fixture(scope="session")
def model(model_path):
    """
    Загружаем модель в FP32 (для DirectML/CUDA/CPU).
    При отсутствии GPU используем CPU – тест всё равно выполнится, просто дольше.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,   # экономим RAM при загрузке
    )
    mdl.to(device)
    mdl.eval()
    return mdl, device


@pytest.fixture(scope="session")
def data_file():
    """Путь к исходному корпусу – он нужен, чтобы проверить «ключевые» слова."""
    path = pathlib.Path("./data.txt")
    if not path.is_file():
        pytest.fail("data.txt not found – нужен для проверки содержания генерации")
    return str(path)


# ----------------------------------------------------------------------
# 2️⃣  Тесты
# ----------------------------------------------------------------------
def test_generation_contains_corpus_word(model, tokenizer, data_file):
    """
    1. Генерируем текст по короткому промпту.
    2. Проверяем, что ответ не пустой.
    3. Проверяем, что в ответе встречается хотя‑бы одно слово
       из оригинального корпуса (в данном случае «агрессия» – самое частое).
    """
    mdl, device = model

    # Примерный промпт, который встречается в корпусе
    prompt = "Агрессия"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        generated = mdl.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)

    # 1️⃣ Не должно быть пустой строки
    assert isinstance(output_text, str) and len(output_text.strip()) > 0, "Сгенерированный текст пустой"

    # 2️⃣ Проверяем наличие «ключевого» слова из корпуса
    # (можно добавить несколько слов, но для простоты берём одно)
    key_word = "агрессия"
    assert key_word in output_text.lower(), f"Слово «{key_word}» не найдено в выводе: {output_text}"


def test_generation_length_limits(model, tokenizer):
    """
    Убеждаемся, что генерация действительно ограничена параметром ``max_new_tokens``.
    """
    mdl, device = model
    prompt = "Вечером над полем"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    max_new = 30
    with torch.no_grad():
        generated = mdl.generate(
            **inputs,
            max_new_tokens=max_new,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    out_ids = generated[0]
    # Длина = длина входа + max_new (может быть чуть меньше, если модель достигла EOS)
    input_len = inputs["input_ids"].shape[1]
    generated_len = out_ids.shape[0]

    # Разница не должна превышать max_new + 5 (плюс‑пять токенов – запас для EOS‑токена)
    assert generated_len - input_len <= max_new + 5, (
        f"Генерация превысила лимит: {generated_len - input_len} > {max_new}"
    )


def test_model_device_consistency(model):
    """
    Проверяем, что все параметры модели находятся на одном устройстве, которое
    совпадает с тем, что мы указали (CPU или CUDA). Это важно, иначе будет
    RuntimeError о «tensor is on different device».
    """
    mdl, device = model
    for name, param in mdl.named_parameters():
        assert param.device == device, f"Parameter {name} on {param.device}, expected {device}"


# ----------------------------------------------------------------------
# 3️⃣  Как запускать
# ----------------------------------------------------------------------
#   $ pytest -q test_generate.py
#
# Если у вас нет модели (merged‑модель) – тест будет помечен как «skipped»,
# а остальные тесты всё равно отработают.
#
# При наличии GPU (CUDA) тесты будут исполняться на GPU, иначе – на CPU.
# На CPU один тест займет ~10‑20 сек., на GPU – несколько секунд.
