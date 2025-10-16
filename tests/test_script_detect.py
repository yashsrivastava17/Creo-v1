from jarvis.lang.script_detect import detect_lang_from_text


def test_detect_english_sentence() -> None:
    assert detect_lang_from_text("How are you?") == "en"


def test_detect_hindi_sentence() -> None:
    assert detect_lang_from_text("तुम कैसे हो?") == "hi"


def test_detect_hinglish_latin() -> None:
    assert detect_lang_from_text("kal subah milte hain ok?") == "en"


def test_detect_mixed_devanagari() -> None:
    assert detect_lang_from_text("भैया ok?") == "hi"
