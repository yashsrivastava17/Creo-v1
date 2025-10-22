from __future__ import annotations

import pytest

from jarvis.tts.voice_router import load_router


def fresh_router():
    load_router.cache_clear()
    return load_router()


def test_female_defaults_en() -> None:
    router = fresh_router()
    weights, variant = router.resolve("female", None, "en")
    assert variant == "normal"
    assert pytest.approx(weights["hf_alpha"]) == 0.4
    assert pytest.approx(weights["af_sky"]) == 1.0


def test_female_defaults_hi() -> None:
    router = fresh_router()
    weights, variant = router.resolve("female", None, "hi")
    assert variant == "normal"
    assert pytest.approx(weights["hf_alpha"]) == 1.2
    assert pytest.approx(weights["af_sky"]) == 0.7


def test_male_alt2_hi() -> None:
    router = fresh_router()
    weights, variant = router.resolve("male", "alt2", "hi")
    assert variant == "alt2"
    assert pytest.approx(weights["hm_omega"]) == 1.3
    assert pytest.approx(weights["am_michael"]) == 1.0


def test_special_variant_any_language() -> None:
    router = fresh_router()
    weights, variant = router.resolve("female", "special_weird", "en")
    assert variant == "special_weird"
    assert weights["voice"] == "pf_dora"


def test_camera_event_blackie() -> None:
    router = fresh_router()
    cfg = router.camera_event("see_blackie")
    assert cfg == {"persona": "female", "variant": "anime", "line": "blackieee"}
