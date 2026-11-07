
import speech_recognition as sr
from pydub import AudioSegment


def get_audio_metadata(audio_path):
    """
    Extract audio sample rate and channel count.
    """
    audio = AudioSegment.from_file(
        str(audio_path),
        format="wav"
    )

    frame_rate = audio.frame_rate
    number_channels = audio.channels

    return frame_rate, number_channels


def transcribe_audio(audio_path, language="en-US"):
    """
    Convert an audio file into text using Google Speech Recognition.
    Requires an internet connection.
    """
    recognizer = sr.Recognizer()

    with sr.AudioFile(str(audio_path)) as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(
            audio_data,
            language=language
        )
        return text

    except sr.UnknownValueError:
        return ""

    except sr.RequestError as error:
        raise RuntimeError(
            f"Speech recognition service error: {error}"
        ) from error
