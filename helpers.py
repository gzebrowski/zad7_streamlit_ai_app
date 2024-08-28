from hashlib import md5
from io import BytesIO
from typing import Any, Union

import streamlit as st

AUDIO_TRANSCRIBE_MODEL = "whisper-1"


def write_html(html: str):
    st.write(html, unsafe_allow_html=True)


def to_int(val: Union[str, int, float], default: Any = None) -> int | None:
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def transcribe_audio(openai_client, audio_bytes: bytes, language='pl') -> str:
    audio_file = BytesIO(audio_bytes)
    audio_file.name = "audio.mp3"
    transcript = openai_client.audio.transcriptions.create(
        file=audio_file,
        model=AUDIO_TRANSCRIBE_MODEL,
        response_format="verbose_json",
        language=language,
    )
    return transcript.text


def get_mp3_audio_and_hash(rec_data):
    audio = BytesIO()
    rec_data.export(audio, format="mp3")
    audio_bytes = audio.getvalue()
    hsh = md5(audio_bytes).hexdigest()
    return audio_bytes, hsh


def is_numeric(val):
    try:
        if val is not None:
            float(val)
    except (TypeError, ValueError):
        return False
    return True

'''
# TODO
def text_to_number(txt: str) -> str:
    first_numbers = ['jeden', 'dwa', 'trzy', 'cztery', 'pięć', 'sześć', 'siedem', 'osiem', 'dziewięć', 'dziesięć',
                     'jedenaście', 'dwanaście', 'trzynaście', 'czternaście', 'piętnaście', 'szesnaście', 'siedemnaście',
                     'osiemnaście', 'dziewiętnaście', 'dwadzieścia']
    trans_map = {fn: nr + 1 for nr, fn in enumerate(first_numbers)}
    decimals = ['trzydzieści', 'czterdzieści', 'pięćdziesiąt', 'sześćdziesiąt', 'siedemdziesiąt', 'osiemdziesiąt',
                'dziewięćdziesiąt']
    trans_map.update({dc: 30 + 10 * nr for nr, dc in enumerate(decimals)})
'''
