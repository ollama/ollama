#!/usr/bin/env python3

"""Library for performing speech recognition, with support for several engines and APIs, online and offline."""

import io
import os
import tempfile
import sys
import subprocess
import wave
import aifc
import math
import audioop
import collections
import json
import base64
import threading
import hashlib
import hmac
import time
import uuid

try:
    import requests
except (ModuleNotFoundError, ImportError):
    pass

__author__ = "Anthony Zhang (Uberi)"
__version__ = "3.10.0"
__license__ = "BSD"

from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from .audio import AudioData, get_flac_converter
from .exceptions import (
    RequestError,
    TranscriptionFailed, 
    TranscriptionNotReady,
    UnknownValueError,
    WaitTimeoutError,
)
from .recognizers import whisper


class AudioSource(object):
    def __init__(self):
        raise NotImplementedError("this is an abstract class")

    def __enter__(self):
        raise NotImplementedError("this is an abstract class")

    def __exit__(self, exc_type, exc_value, traceback):
        raise NotImplementedError("this is an abstract class")


class Microphone(AudioSource):
    """
    Creates a new ``Microphone`` instance, which represents a physical microphone on the computer. Subclass of ``AudioSource``.

    This will throw an ``AttributeError`` if you don't have PyAudio 0.2.11 or later installed.

    If ``device_index`` is unspecified or ``None``, the default microphone is used as the audio source. Otherwise, ``device_index`` should be the index of the device to use for audio input.

    A device index is an integer between 0 and ``pyaudio.get_device_count() - 1`` (assume we have used ``import pyaudio`` beforehand) inclusive. It represents an audio device such as a microphone or speaker. See the `PyAudio documentation <http://people.csail.mit.edu/hubert/pyaudio/docs/>`__ for more details.

    The microphone audio is recorded in chunks of ``chunk_size`` samples, at a rate of ``sample_rate`` samples per second (Hertz). If not specified, the value of ``sample_rate`` is determined automatically from the system's microphone settings.

    Higher ``sample_rate`` values result in better audio quality, but also more bandwidth (and therefore, slower recognition). Additionally, some CPUs, such as those in older Raspberry Pi models, can't keep up if this value is too high.

    Higher ``chunk_size`` values help avoid triggering on rapidly changing ambient noise, but also makes detection less sensitive. This value, generally, should be left at its default.
    """
    def __init__(self, device_index=None, sample_rate=None, chunk_size=1024):
        assert device_index is None or isinstance(device_index, int), "Device index must be None or an integer"
        assert sample_rate is None or (isinstance(sample_rate, int) and sample_rate > 0), "Sample rate must be None or a positive integer"
        assert isinstance(chunk_size, int) and chunk_size > 0, "Chunk size must be a positive integer"

        # set up PyAudio
        self.pyaudio_module = self.get_pyaudio()
        audio = self.pyaudio_module.PyAudio()
        try:
            count = audio.get_device_count()  # obtain device count
            if device_index is not None:  # ensure device index is in range
                assert 0 <= device_index < count, "Device index out of range ({} devices available; device index should be between 0 and {} inclusive)".format(count, count - 1)
            if sample_rate is None:  # automatically set the sample rate to the hardware's default sample rate if not specified
                device_info = audio.get_device_info_by_index(device_index) if device_index is not None else audio.get_default_input_device_info()
                assert isinstance(device_info.get("defaultSampleRate"), (float, int)) and device_info["defaultSampleRate"] > 0, "Invalid device info returned from PyAudio: {}".format(device_info)
                sample_rate = int(device_info["defaultSampleRate"])
        finally:
            audio.terminate()

        self.device_index = device_index
        self.format = self.pyaudio_module.paInt16  # 16-bit int sampling
        self.SAMPLE_WIDTH = self.pyaudio_module.get_sample_size(self.format)  # size of each sample
        self.SAMPLE_RATE = sample_rate  # sampling rate in Hertz
        self.CHUNK = chunk_size  # number of frames stored in each buffer

        self.audio = None
        self.stream = None

    @staticmethod
    def get_pyaudio():
        """
        Imports the pyaudio module and checks its version. Throws exceptions if pyaudio can't be found or a wrong version is installed
        """
        try:
            import pyaudio
        except ImportError:
            raise AttributeError("Could not find PyAudio; check installation")
        from distutils.version import LooseVersion
        if LooseVersion(pyaudio.__version__) < LooseVersion("0.2.11"):
            raise AttributeError("PyAudio 0.2.11 or later is required (found version {})".format(pyaudio.__version__))
        return pyaudio

    @staticmethod
    def list_microphone_names():
        """
        Returns a list of the names of all available microphones. For microphones where the name can't be retrieved, the list entry contains ``None`` instead.

        The index of each microphone's name in the returned list is the same as its device index when creating a ``Microphone`` instance - if you want to use the microphone at index 3 in the returned list, use ``Microphone(device_index=3)``.
        """
        audio = Microphone.get_pyaudio().PyAudio()
        try:
            result = []
            for i in range(audio.get_device_count()):
                device_info = audio.get_device_info_by_index(i)
                result.append(device_info.get("name"))
        finally:
            audio.terminate()
        return result

    @staticmethod
    def list_working_microphones():
        """
        Returns a dictionary mapping device indices to microphone names, for microphones that are currently hearing sounds. When using this function, ensure that your microphone is unmuted and make some noise at it to ensure it will be detected as working.

        Each key in the returned dictionary can be passed to the ``Microphone`` constructor to use that microphone. For example, if the return value is ``{3: "HDA Intel PCH: ALC3232 Analog (hw:1,0)"}``, you can do ``Microphone(device_index=3)`` to use that microphone.
        """
        pyaudio_module = Microphone.get_pyaudio()
        audio = pyaudio_module.PyAudio()
        try:
            result = {}
            for device_index in range(audio.get_device_count()):
                device_info = audio.get_device_info_by_index(device_index)
                device_name = device_info.get("name")
                assert isinstance(device_info.get("defaultSampleRate"), (float, int)) and device_info["defaultSampleRate"] > 0, "Invalid device info returned from PyAudio: {}".format(device_info)
                try:
                    # read audio
                    pyaudio_stream = audio.open(
                        input_device_index=device_index, channels=1, format=pyaudio_module.paInt16,
                        rate=int(device_info["defaultSampleRate"]), input=True
                    )
                    try:
                        buffer = pyaudio_stream.read(1024)
                        if not pyaudio_stream.is_stopped(): pyaudio_stream.stop_stream()
                    finally:
                        pyaudio_stream.close()
                except Exception:
                    continue

                # compute RMS of debiased audio
                energy = -audioop.rms(buffer, 2)
                energy_bytes = bytes([energy & 0xFF, (energy >> 8) & 0xFF])
                debiased_energy = audioop.rms(audioop.add(buffer, energy_bytes * (len(buffer) // 2), 2), 2)

                if debiased_energy > 30:  # probably actually audio
                    result[device_index] = device_name
        finally:
            audio.terminate()
        return result

    def __enter__(self):
        assert self.stream is None, "This audio source is already inside a context manager"
        self.audio = self.pyaudio_module.PyAudio()
        try:
            self.stream = Microphone.MicrophoneStream(
                self.audio.open(
                    input_device_index=self.device_index, channels=1, format=self.format,
                    rate=self.SAMPLE_RATE, frames_per_buffer=self.CHUNK, input=True,
                )
            )
        except Exception:
            self.audio.terminate()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.stream.close()
        finally:
            self.stream = None
            self.audio.terminate()

    class MicrophoneStream(object):
        def __init__(self, pyaudio_stream):
            self.pyaudio_stream = pyaudio_stream

        def read(self, size):
            return self.pyaudio_stream.read(size, exception_on_overflow=False)

        def close(self):
            try:
                # sometimes, if the stream isn't stopped, closing the stream throws an exception
                if not self.pyaudio_stream.is_stopped():
                    self.pyaudio_stream.stop_stream()
            finally:
                self.pyaudio_stream.close()


class AudioFile(AudioSource):
    """
    Creates a new ``AudioFile`` instance given a WAV/AIFF/FLAC audio file ``filename_or_fileobject``. Subclass of ``AudioSource``.

    If ``filename_or_fileobject`` is a string, then it is interpreted as a path to an audio file on the filesystem. Otherwise, ``filename_or_fileobject`` should be a file-like object such as ``io.BytesIO`` or similar.

    Note that functions that read from the audio (such as ``recognizer_instance.record`` or ``recognizer_instance.listen``) will move ahead in the stream. For example, if you execute ``recognizer_instance.record(audiofile_instance, duration=10)`` twice, the first time it will return the first 10 seconds of audio, and the second time it will return the 10 seconds of audio right after that. This is always reset to the beginning when entering an ``AudioFile`` context.

    WAV files must be in PCM/LPCM format; WAVE_FORMAT_EXTENSIBLE and compressed WAV are not supported and may result in undefined behaviour.

    Both AIFF and AIFF-C (compressed AIFF) formats are supported.

    FLAC files must be in native FLAC format; OGG-FLAC is not supported and may result in undefined behaviour.
    """

    def __init__(self, filename_or_fileobject):
        assert isinstance(filename_or_fileobject, (type(""), type(u""))) or hasattr(filename_or_fileobject, "read"), "Given audio file must be a filename string or a file-like object"
        self.filename_or_fileobject = filename_or_fileobject
        self.stream = None
        self.DURATION = None

        self.audio_reader = None
        self.little_endian = False
        self.SAMPLE_RATE = None
        self.CHUNK = None
        self.FRAME_COUNT = None

    def __enter__(self):
        assert self.stream is None, "This audio source is already inside a context manager"
        try:
            # attempt to read the file as WAV
            self.audio_reader = wave.open(self.filename_or_fileobject, "rb")
            self.little_endian = True  # RIFF WAV is a little-endian format (most ``audioop`` operations assume that the frames are stored in little-endian form)
        except (wave.Error, EOFError):
            try:
                # attempt to read the file as AIFF
                self.audio_reader = aifc.open(self.filename_or_fileobject, "rb")
                self.little_endian = False  # AIFF is a big-endian format
            except (aifc.Error, EOFError):
                # attempt to read the file as FLAC
                if hasattr(self.filename_or_fileobject, "read"):
                    flac_data = self.filename_or_fileobject.read()
                else:
                    with open(self.filename_or_fileobject, "rb") as f: flac_data = f.read()

                # run the FLAC converter with the FLAC data to get the AIFF data
                flac_converter = get_flac_converter()
                if os.name == "nt":  # on Windows, specify that the process is to be started without showing a console window
                    startup_info = subprocess.STARTUPINFO()
                    startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # specify that the wShowWindow field of `startup_info` contains a value
                    startup_info.wShowWindow = subprocess.SW_HIDE  # specify that the console window should be hidden
                else:
                    startup_info = None  # default startupinfo
                process = subprocess.Popen([
                    flac_converter,
                    "--stdout", "--totally-silent",  # put the resulting AIFF file in stdout, and make sure it's not mixed with any program output
                    "--decode", "--force-aiff-format",  # decode the FLAC file into an AIFF file
                    "-",  # the input FLAC file contents will be given in stdin
                ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, startupinfo=startup_info)
                aiff_data, _ = process.communicate(flac_data)
                aiff_file = io.BytesIO(aiff_data)
                try:
                    self.audio_reader = aifc.open(aiff_file, "rb")
                except (aifc.Error, EOFError):
                    raise ValueError("Audio file could not be read as PCM WAV, AIFF/AIFF-C, or Native FLAC; check if file is corrupted or in another format")
                self.little_endian = False  # AIFF is a big-endian format
        assert 1 <= self.audio_reader.getnchannels() <= 2, "Audio must be mono or stereo"
        self.SAMPLE_WIDTH = self.audio_reader.getsampwidth()

        # 24-bit audio needs some special handling for old Python versions (workaround for https://bugs.python.org/issue12866)
        samples_24_bit_pretending_to_be_32_bit = False
        if self.SAMPLE_WIDTH == 3:  # 24-bit audio
            try: audioop.bias(b"", self.SAMPLE_WIDTH, 0)  # test whether this sample width is supported (for example, ``audioop`` in Python 3.3 and below don't support sample width 3, while Python 3.4+ do)
            except audioop.error:  # this version of audioop doesn't support 24-bit audio (probably Python 3.3 or less)
                samples_24_bit_pretending_to_be_32_bit = True  # while the ``AudioFile`` instance will outwardly appear to be 32-bit, it will actually internally be 24-bit
                self.SAMPLE_WIDTH = 4  # the ``AudioFile`` instance should present itself as a 32-bit stream now, since we'll be converting into 32-bit on the fly when reading

        self.SAMPLE_RATE = self.audio_reader.getframerate()
        self.CHUNK = 4096
        self.FRAME_COUNT = self.audio_reader.getnframes()
        self.DURATION = self.FRAME_COUNT / float(self.SAMPLE_RATE)
        self.stream = AudioFile.AudioFileStream(self.audio_reader, self.little_endian, samples_24_bit_pretending_to_be_32_bit)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not hasattr(self.filename_or_fileobject, "read"):  # only close the file if it was opened by this class in the first place (if the file was originally given as a path)
            self.audio_reader.close()
        self.stream = None
        self.DURATION = None

    class AudioFileStream(object):
        def __init__(self, audio_reader, little_endian, samples_24_bit_pretending_to_be_32_bit):
            self.audio_reader = audio_reader  # an audio file object (e.g., a `wave.Wave_read` instance)
            self.little_endian = little_endian  # whether the audio data is little-endian (when working with big-endian things, we'll have to convert it to little-endian before we process it)
            self.samples_24_bit_pretending_to_be_32_bit = samples_24_bit_pretending_to_be_32_bit  # this is true if the audio is 24-bit audio, but 24-bit audio isn't supported, so we have to pretend that this is 32-bit audio and convert it on the fly

        def read(self, size=-1):
            buffer = self.audio_reader.readframes(self.audio_reader.getnframes() if size == -1 else size)
            if not isinstance(buffer, bytes): buffer = b""  # workaround for https://bugs.python.org/issue24608

            sample_width = self.audio_reader.getsampwidth()
            if not self.little_endian:  # big endian format, convert to little endian on the fly
                if hasattr(audioop, "byteswap"):  # ``audioop.byteswap`` was only added in Python 3.4 (incidentally, that also means that we don't need to worry about 24-bit audio being unsupported, since Python 3.4+ always has that functionality)
                    buffer = audioop.byteswap(buffer, sample_width)
                else:  # manually reverse the bytes of each sample, which is slower but works well enough as a fallback
                    buffer = buffer[sample_width - 1::-1] + b"".join(buffer[i + sample_width:i:-1] for i in range(sample_width - 1, len(buffer), sample_width))

            # workaround for https://bugs.python.org/issue12866
            if self.samples_24_bit_pretending_to_be_32_bit:  # we need to convert samples from 24-bit to 32-bit before we can process them with ``audioop`` functions
                buffer = b"".join(b"\x00" + buffer[i:i + sample_width] for i in range(0, len(buffer), sample_width))  # since we're in little endian, we prepend a zero byte to each 24-bit sample to get a 32-bit sample
                sample_width = 4  # make sure we thread the buffer as 32-bit audio now, after converting it from 24-bit audio
            if self.audio_reader.getnchannels() != 1:  # stereo audio
                buffer = audioop.tomono(buffer, sample_width, 1, 1)  # convert stereo audio data to mono
            return buffer


class Recognizer(AudioSource):
    def __init__(self):
        """
        Creates a new ``Recognizer`` instance, which represents a collection of speech recognition functionality.
        """
        self.energy_threshold = 300  # minimum audio energy to consider for recording
        self.dynamic_energy_threshold = True
        self.dynamic_energy_adjustment_damping = 0.15
        self.dynamic_energy_ratio = 1.5
        self.pause_threshold = 0.8  # seconds of non-speaking audio before a phrase is considered complete
        self.operation_timeout = None  # seconds after an internal operation (e.g., an API request) starts before it times out, or ``None`` for no timeout

        self.phrase_threshold = 0.3  # minimum seconds of speaking audio before we consider the speaking audio a phrase - values below this are ignored (for filtering out clicks and pops)
        self.non_speaking_duration = 0.5  # seconds of non-speaking audio to keep on both sides of the recording

    def record(self, source, duration=None, offset=None):
        """
        Records up to ``duration`` seconds of audio from ``source`` (an ``AudioSource`` instance) starting at ``offset`` (or at the beginning if not specified) into an ``AudioData`` instance, which it returns.

        If ``duration`` is not specified, then it will record until there is no more audio input.
        """
        assert isinstance(source, AudioSource), "Source must be an audio source"
        assert source.stream is not None, "Audio source must be entered before recording, see documentation for ``AudioSource``; are you using ``source`` outside of a ``with`` statement?"

        frames = io.BytesIO()
        seconds_per_buffer = (source.CHUNK + 0.0) / source.SAMPLE_RATE
        elapsed_time = 0
        offset_time = 0
        offset_reached = False
        while True:  # loop for the total number of chunks needed
            if offset and not offset_reached:
                offset_time += seconds_per_buffer
                if offset_time > offset:
                    offset_reached = True

            buffer = source.stream.read(source.CHUNK)
            if len(buffer) == 0: break

            if offset_reached or not offset:
                elapsed_time += seconds_per_buffer
                if duration and elapsed_time > duration: break

                frames.write(buffer)

        frame_data = frames.getvalue()
        frames.close()
        return AudioData(frame_data, source.SAMPLE_RATE, source.SAMPLE_WIDTH)

    def adjust_for_ambient_noise(self, source, duration=1):
        """
        Adjusts the energy threshold dynamically using audio from ``source`` (an ``AudioSource`` instance) to account for ambient noise.

        Intended to calibrate the energy threshold with the ambient energy level. Should be used on periods of audio without speech - will stop early if any speech is detected.

        The ``duration`` parameter is the maximum number of seconds that it will dynamically adjust the threshold for before returning. This value should be at least 0.5 in order to get a representative sample of the ambient noise.
        """
        assert isinstance(source, AudioSource), "Source must be an audio source"
        assert source.stream is not None, "Audio source must be entered before adjusting, see documentation for ``AudioSource``; are you using ``source`` outside of a ``with`` statement?"
        assert self.pause_threshold >= self.non_speaking_duration >= 0

        seconds_per_buffer = (source.CHUNK + 0.0) / source.SAMPLE_RATE
        elapsed_time = 0

        # adjust energy threshold until a phrase starts
        while True:
            elapsed_time += seconds_per_buffer
            if elapsed_time > duration: break
            buffer = source.stream.read(source.CHUNK)
            energy = audioop.rms(buffer, source.SAMPLE_WIDTH)  # energy of the audio signal

            # dynamically adjust the energy threshold using asymmetric weighted average
            damping = self.dynamic_energy_adjustment_damping ** seconds_per_buffer  # account for different chunk sizes and rates
            target_energy = energy * self.dynamic_energy_ratio
            self.energy_threshold = self.energy_threshold * damping + target_energy * (1 - damping)

    def snowboy_wait_for_hot_word(self, snowboy_location, snowboy_hot_word_files, source, timeout=None):
        # load snowboy library (NOT THREAD SAFE)
        sys.path.append(snowboy_location)
        import snowboydetect
        sys.path.pop()

        detector = snowboydetect.SnowboyDetect(
            resource_filename=os.path.join(snowboy_location, "resources", "common.res").encode(),
            model_str=",".join(snowboy_hot_word_files).encode()
        )
        detector.SetAudioGain(1.0)
        detector.SetSensitivity(",".join(["0.4"] * len(snowboy_hot_word_files)).encode())
        snowboy_sample_rate = detector.SampleRate()

        elapsed_time = 0
        seconds_per_buffer = float(source.CHUNK) / source.SAMPLE_RATE
        resampling_state = None

        # buffers capable of holding 5 seconds of original audio
        five_seconds_buffer_count = int(math.ceil(5 / seconds_per_buffer))
        # buffers capable of holding 0.5 seconds of resampled audio
        half_second_buffer_count = int(math.ceil(0.5 / seconds_per_buffer))
        frames = collections.deque(maxlen=five_seconds_buffer_count)
        resampled_frames = collections.deque(maxlen=half_second_buffer_count)
        # snowboy check interval
        check_interval = 0.05
        last_check = time.time()
        while True:
            elapsed_time += seconds_per_buffer
            if timeout and elapsed_time > timeout:
                raise WaitTimeoutError("listening timed out while waiting for hotword to be said")

            buffer = source.stream.read(source.CHUNK)
            if len(buffer) == 0: break  # reached end of the stream
            frames.append(buffer)

            # resample audio to the required sample rate
            resampled_buffer, resampling_state = audioop.ratecv(buffer, source.SAMPLE_WIDTH, 1, source.SAMPLE_RATE, snowboy_sample_rate, resampling_state)
            resampled_frames.append(resampled_buffer)
            if time.time() - last_check > check_interval:
                # run Snowboy on the resampled audio
                snowboy_result = detector.RunDetection(b"".join(resampled_frames))
                assert snowboy_result != -1, "Error initializing streams or reading audio data"
                if snowboy_result > 0: break  # wake word found
                resampled_frames.clear()
                last_check = time.time()

        return b"".join(frames), elapsed_time

    def listen(self, source, timeout=None, phrase_time_limit=None, snowboy_configuration=None):
        """
        Records a single phrase from ``source`` (an ``AudioSource`` instance) into an ``AudioData`` instance, which it returns.

        This is done by waiting until the audio has an energy above ``recognizer_instance.energy_threshold`` (the user has started speaking), and then recording until it encounters ``recognizer_instance.pause_threshold`` seconds of non-speaking or there is no more audio input. The ending silence is not included.

        The ``timeout`` parameter is the maximum number of seconds that this will wait for a phrase to start before giving up and throwing an ``speech_recognition.WaitTimeoutError`` exception. If ``timeout`` is ``None``, there will be no wait timeout.

        The ``phrase_time_limit`` parameter is the maximum number of seconds that this will allow a phrase to continue before stopping and returning the part of the phrase processed before the time limit was reached. The resulting audio will be the phrase cut off at the time limit. If ``phrase_timeout`` is ``None``, there will be no phrase time limit.

        The ``snowboy_configuration`` parameter allows integration with `Snowboy <https://snowboy.kitt.ai/>`__, an offline, high-accuracy, power-efficient hotword recognition engine. When used, this function will pause until Snowboy detects a hotword, after which it will unpause. This parameter should either be ``None`` to turn off Snowboy support, or a tuple of the form ``(SNOWBOY_LOCATION, LIST_OF_HOT_WORD_FILES)``, where ``SNOWBOY_LOCATION`` is the path to the Snowboy root directory, and ``LIST_OF_HOT_WORD_FILES`` is a list of paths to Snowboy hotword configuration files (`*.pmdl` or `*.umdl` format).

        This operation will always complete within ``timeout + phrase_timeout`` seconds if both are numbers, either by returning the audio data, or by raising a ``speech_recognition.WaitTimeoutError`` exception.
        """
        assert isinstance(source, AudioSource), "Source must be an audio source"
        assert source.stream is not None, "Audio source must be entered before listening, see documentation for ``AudioSource``; are you using ``source`` outside of a ``with`` statement?"
        assert self.pause_threshold >= self.non_speaking_duration >= 0
        if snowboy_configuration is not None:
            assert os.path.isfile(os.path.join(snowboy_configuration[0], "snowboydetect.py")), "``snowboy_configuration[0]`` must be a Snowboy root directory containing ``snowboydetect.py``"
            for hot_word_file in snowboy_configuration[1]:
                assert os.path.isfile(hot_word_file), "``snowboy_configuration[1]`` must be a list of Snowboy hot word configuration files"

        seconds_per_buffer = float(source.CHUNK) / source.SAMPLE_RATE
        pause_buffer_count = int(math.ceil(self.pause_threshold / seconds_per_buffer))  # number of buffers of non-speaking audio during a phrase, before the phrase should be considered complete
        phrase_buffer_count = int(math.ceil(self.phrase_threshold / seconds_per_buffer))  # minimum number of buffers of speaking audio before we consider the speaking audio a phrase
        non_speaking_buffer_count = int(math.ceil(self.non_speaking_duration / seconds_per_buffer))  # maximum number of buffers of non-speaking audio to retain before and after a phrase

        # read audio input for phrases until there is a phrase that is long enough
        elapsed_time = 0  # number of seconds of audio read
        buffer = b""  # an empty buffer means that the stream has ended and there is no data left to read
        while True:
            frames = collections.deque()

            if snowboy_configuration is None:
                # store audio input until the phrase starts
                while True:
                    # handle waiting too long for phrase by raising an exception
                    elapsed_time += seconds_per_buffer
                    if timeout and elapsed_time > timeout:
                        raise WaitTimeoutError("listening timed out while waiting for phrase to start")

                    buffer = source.stream.read(source.CHUNK)
                    if len(buffer) == 0: break  # reached end of the stream
                    frames.append(buffer)
                    if len(frames) > non_speaking_buffer_count:  # ensure we only keep the needed amount of non-speaking buffers
                        frames.popleft()

                    # detect whether speaking has started on audio input
                    energy = audioop.rms(buffer, source.SAMPLE_WIDTH)  # energy of the audio signal
                    if energy > self.energy_threshold: break

                    # dynamically adjust the energy threshold using asymmetric weighted average
                    if self.dynamic_energy_threshold:
                        damping = self.dynamic_energy_adjustment_damping ** seconds_per_buffer  # account for different chunk sizes and rates
                        target_energy = energy * self.dynamic_energy_ratio
                        self.energy_threshold = self.energy_threshold * damping + target_energy * (1 - damping)
            else:
                # read audio input until the hotword is said
                snowboy_location, snowboy_hot_word_files = snowboy_configuration
                buffer, delta_time = self.snowboy_wait_for_hot_word(snowboy_location, snowboy_hot_word_files, source, timeout)
                elapsed_time += delta_time
                if len(buffer) == 0: break  # reached end of the stream
                frames.append(buffer)

            # read audio input until the phrase ends
            pause_count, phrase_count = 0, 0
            phrase_start_time = elapsed_time
            while True:
                # handle phrase being too long by cutting off the audio
                elapsed_time += seconds_per_buffer
                if phrase_time_limit and elapsed_time - phrase_start_time > phrase_time_limit:
                    break

                buffer = source.stream.read(source.CHUNK)
                if len(buffer) == 0: break  # reached end of the stream
                frames.append(buffer)
                phrase_count += 1

                # check if speaking has stopped for longer than the pause threshold on the audio input
                energy = audioop.rms(buffer, source.SAMPLE_WIDTH)  # unit energy of the audio signal within the buffer
                if energy > self.energy_threshold:
                    pause_count = 0
                else:
                    pause_count += 1
                if pause_count > pause_buffer_count:  # end of the phrase
                    break

            # check how long the detected phrase is, and retry listening if the phrase is too short
            phrase_count -= pause_count  # exclude the buffers for the pause before the phrase
            if phrase_count >= phrase_buffer_count or len(buffer) == 0: break  # phrase is long enough or we've reached the end of the stream, so stop listening

        # obtain frame data
        for i in range(pause_count - non_speaking_buffer_count): frames.pop()  # remove extra non-speaking frames at the end
        frame_data = b"".join(frames)

        return AudioData(frame_data, source.SAMPLE_RATE, source.SAMPLE_WIDTH)

    def listen_in_background(self, source, callback, phrase_time_limit=None):
        """
        Spawns a thread to repeatedly record phrases from ``source`` (an ``AudioSource`` instance) into an ``AudioData`` instance and call ``callback`` with that ``AudioData`` instance as soon as each phrase are detected.

        Returns a function object that, when called, requests that the background listener thread stop. The background thread is a daemon and will not stop the program from exiting if there are no other non-daemon threads. The function accepts one parameter, ``wait_for_stop``: if truthy, the function will wait for the background listener to stop before returning, otherwise it will return immediately and the background listener thread might still be running for a second or two afterwards. Additionally, if you are using a truthy value for ``wait_for_stop``, you must call the function from the same thread you originally called ``listen_in_background`` from.

        Phrase recognition uses the exact same mechanism as ``recognizer_instance.listen(source)``. The ``phrase_time_limit`` parameter works in the same way as the ``phrase_time_limit`` parameter for ``recognizer_instance.listen(source)``, as well.

        The ``callback`` parameter is a function that should accept two parameters - the ``recognizer_instance``, and an ``AudioData`` instance representing the captured audio. Note that ``callback`` function will be called from a non-main thread.
        """
        assert isinstance(source, AudioSource), "Source must be an audio source"
        running = [True]

        def threaded_listen():
            with source as s:
                while running[0]:
                    try:  # listen for 1 second, then check again if the stop function has been called
                        audio = self.listen(s, 1, phrase_time_limit)
                    except WaitTimeoutError:  # listening timed out, just try again
                        pass
                    else:
                        if running[0]: callback(self, audio)

        def stopper(wait_for_stop=True):
            running[0] = False
            if wait_for_stop:
                listener_thread.join()  # block until the background thread is done, which can take around 1 second

        listener_thread = threading.Thread(target=threaded_listen)
        listener_thread.daemon = True
        listener_thread.start()
        return stopper

    def recognize_sphinx(self, audio_data, language="en-US", keyword_entries=None, grammar=None, show_all=False):
        """
        Performs speech recognition on ``audio_data`` (an ``AudioData`` instance), using CMU Sphinx.

        The recognition language is determined by ``language``, an RFC5646 language tag like ``"en-US"`` or ``"en-GB"``, defaulting to US English. Out of the box, only ``en-US`` is supported. See `Notes on using `PocketSphinx <https://github.com/Uberi/speech_recognition/blob/master/reference/pocketsphinx.rst>`__ for information about installing other languages. This document is also included under ``reference/pocketsphinx.rst``. The ``language`` parameter can also be a tuple of filesystem paths, of the form ``(acoustic_parameters_directory, language_model_file, phoneme_dictionary_file)`` - this allows you to load arbitrary Sphinx models.

        If specified, the keywords to search for are determined by ``keyword_entries``, an iterable of tuples of the form ``(keyword, sensitivity)``, where ``keyword`` is a phrase, and ``sensitivity`` is how sensitive to this phrase the recognizer should be, on a scale of 0 (very insensitive, more false negatives) to 1 (very sensitive, more false positives) inclusive. If not specified or ``None``, no keywords are used and Sphinx will simply transcribe whatever words it recognizes. Specifying ``keyword_entries`` is more accurate than just looking for those same keywords in non-keyword-based transcriptions, because Sphinx knows specifically what sounds to look for.

        Sphinx can also handle FSG or JSGF grammars. The parameter ``grammar`` expects a path to the grammar file. Note that if a JSGF grammar is passed, an FSG grammar will be created at the same location to speed up execution in the next run. If ``keyword_entries`` are passed, content of ``grammar`` will be ignored.

        Returns the most likely transcription if ``show_all`` is false (the default). Otherwise, returns the Sphinx ``pocketsphinx.pocketsphinx.Decoder`` object resulting from the recognition.

        Raises a ``speech_recognition.UnknownValueError`` exception if the speech is unintelligible. Raises a ``speech_recognition.RequestError`` exception if there are any issues with the Sphinx installation.
        """
        assert isinstance(audio_data, AudioData), "``audio_data`` must be audio data"
        assert isinstance(language, str) or (isinstance(language, tuple) and len(language) == 3), "``language`` must be a string or 3-tuple of Sphinx data file paths of the form ``(acoustic_parameters, language_model, phoneme_dictionary)``"
        assert keyword_entries is None or all(isinstance(keyword, (type(""), type(u""))) and 0 <= sensitivity <= 1 for keyword, sensitivity in keyword_entries), "``keyword_entries`` must be ``None`` or a list of pairs of strings and numbers between 0 and 1"

        # import the PocketSphinx speech recognition module
        try:
            from pocketsphinx import pocketsphinx, Jsgf, FsgModel

        except ImportError:
            raise RequestError("missing PocketSphinx module: ensure that PocketSphinx is set up correctly.")
        except ValueError:
            raise RequestError("bad PocketSphinx installation; try reinstalling PocketSphinx version 0.0.9 or better.")
        if not hasattr(pocketsphinx, "Decoder") or not hasattr(pocketsphinx.Decoder, "default_config"):
            raise RequestError("outdated PocketSphinx installation; ensure you have PocketSphinx version 0.0.9 or better.")

        if isinstance(language, str):  # directory containing language data
            language_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "pocketsphinx-data", language)
            if not os.path.isdir(language_directory):
                raise RequestError("missing PocketSphinx language data directory: \"{}\"".format(language_directory))
            acoustic_parameters_directory = os.path.join(language_directory, "acoustic-model")
            language_model_file = os.path.join(language_directory, "language-model.lm.bin")
            phoneme_dictionary_file = os.path.join(language_directory, "pronounciation-dictionary.dict")
        else:  # 3-tuple of Sphinx data file paths
            acoustic_parameters_directory, language_model_file, phoneme_dictionary_file = language
        if not os.path.isdir(acoustic_parameters_directory):
            raise RequestError("missing PocketSphinx language model parameters directory: \"{}\"".format(acoustic_parameters_directory))
        if not os.path.isfile(language_model_file):
            raise RequestError("missing PocketSphinx language model file: \"{}\"".format(language_model_file))
        if not os.path.isfile(phoneme_dictionary_file):
            raise RequestError("missing PocketSphinx phoneme dictionary file: \"{}\"".format(phoneme_dictionary_file))

        # create decoder object
        config = pocketsphinx.Decoder.default_config()
        config.set_string("-hmm", acoustic_parameters_directory)  # set the path of the hidden Markov model (HMM) parameter files
        config.set_string("-lm", language_model_file)
        config.set_string("-dict", phoneme_dictionary_file)
        config.set_string("-logfn", os.devnull)  # disable logging (logging causes unwanted output in terminal)
        decoder = pocketsphinx.Decoder(config)

        # obtain audio data
        raw_data = audio_data.get_raw_data(convert_rate=16000, convert_width=2)  # the included language models require audio to be 16-bit mono 16 kHz in little-endian format

        # obtain recognition results
        if keyword_entries is not None:  # explicitly specified set of keywords
            with PortableNamedTemporaryFile("w") as f:
                # generate a keywords file - Sphinx documentation recommendeds sensitivities between 1e-50 and 1e-5
                f.writelines("{} /1e{}/\n".format(keyword, 100 * sensitivity - 110) for keyword, sensitivity in keyword_entries)
                f.flush()

                # perform the speech recognition with the keywords file (this is inside the context manager so the file isn;t deleted until we're done)
                decoder.set_kws("keywords", f.name)
                decoder.set_search("keywords")
        elif grammar is not None:  # a path to a FSG or JSGF grammar
            if not os.path.exists(grammar):
                raise ValueError("Grammar '{0}' does not exist.".format(grammar))
            grammar_path = os.path.abspath(os.path.dirname(grammar))
            grammar_name = os.path.splitext(os.path.basename(grammar))[0]
            fsg_path = "{0}/{1}.fsg".format(grammar_path, grammar_name)
            if not os.path.exists(fsg_path):  # create FSG grammar if not available
                jsgf = Jsgf(grammar)
                rule = jsgf.get_rule("{0}.{0}".format(grammar_name))
                fsg = jsgf.build_fsg(rule, decoder.get_logmath(), 7.5)
                fsg.writefile(fsg_path)
            else:
                fsg = FsgModel(fsg_path, decoder.get_logmath(), 7.5)
            decoder.set_fsg(grammar_name, fsg)
            decoder.set_search(grammar_name)

        decoder.start_utt()  # begin utterance processing
        decoder.process_raw(raw_data, False, True)  # process audio data with recognition enabled (no_search = False), as a full utterance (full_utt = True)
        decoder.end_utt()  # stop utterance processing

        if show_all: return decoder

        # return results
        hypothesis = decoder.hyp()
        if hypothesis is not None: return hypothesis.hypstr
        raise UnknownValueError()  # no transcriptions available

    def recognize_google(self, audio_data, key=None, language="en-US", pfilter=0, show_all=False, with_confidence=False):
        """
        Performs speech recognition on ``audio_data`` (an ``AudioData`` instance), using the Google Speech Recognition API.

        The Google Speech Recognition API key is specified by ``key``. If not specified, it uses a generic key that works out of the box. This should generally be used for personal or testing purposes only, as it **may be revoked by Google at any time**.

        To obtain your own API key, simply following the steps on the `API Keys <http://www.chromium.org/developers/how-tos/api-keys>`__ page at the Chromium Developers site. In the Google Developers Console, Google Speech Recognition is listed as "Speech API".

        The recognition language is determined by ``language``, an RFC5646 language tag like ``"en-US"`` (US English) or ``"fr-FR"`` (International French), defaulting to US English. A list of supported language tags can be found in this `StackOverflow answer <http://stackoverflow.com/a/14302134>`__.

        The profanity filter level can be adjusted with ``pfilter``: 0 - No filter, 1 - Only shows the first character and replaces the rest with asterisks. The default is level 0.

        Returns the most likely transcription if ``show_all`` is false (the default). Otherwise, returns the raw API response as a JSON dictionary.

        Raises a ``speech_recognition.UnknownValueError`` exception if the speech is unintelligible. Raises a ``speech_recognition.RequestError`` exception if the speech recognition operation failed, if the key isn't valid, or if there is no internet connection.
        """
        assert isinstance(audio_data, AudioData), "``audio_data`` must be audio data"
        assert key is None or isinstance(key, str), "``key`` must be ``None`` or a string"
        assert isinstance(language, str), "``language`` must be a string"

        flac_data = audio_data.get_flac_data(
            convert_rate=None if audio_data.sample_rate >= 8000 else 8000,  # audio samples must be at least 8 kHz
            convert_width=2  # audio samples must be 16-bit
        )
        if key is None: key = "AIzaSyBOti4mM-6x9WDnZIjIeyEU21OpBXqWBgw"
        url = "http://www.google.com/speech-api/v2/recognize?{}".format(urlencode({
            "client": "chromium",
            "lang": language,
            "key": key,
            "pFilter": pfilter
        }))
        request = Request(url, data=flac_data, headers={"Content-Type": "audio/x-flac; rate={}".format(audio_data.sample_rate)})

        # obtain audio transcription results
        try:
            response = urlopen(request, timeout=self.operation_timeout)
        except HTTPError as e:
            raise RequestError("recognition request failed: {}".format(e.reason))
        except URLError as e:
            raise RequestError("recognition connection failed: {}".format(e.reason))
        response_text = response.read().decode("utf-8")

        # ignore any blank blocks
        actual_result = []
        for line in response_text.split("\n"):
            if not line: continue
            result = json.loads(line)["result"]
            if len(result) != 0:
                actual_result = result[0]
                break

        # return results
        if show_all:
            return actual_result

        if not isinstance(actual_result, dict) or len(actual_result.get("alternative", [])) == 0: raise UnknownValueError()

        if "confidence" in actual_result["alternative"]:
            # return alternative with highest confidence score
            best_hypothesis = max(actual_result["alternative"], key=lambda alternative: alternative["confidence"])
        else:
            # when there is no confidence available, we arbitrarily choose the first hypothesis.
            best_hypothesis = actual_result["alternative"][0]
        if "transcript" not in best_hypothesis: raise UnknownValueError()
        # https://cloud.google.com/speech-to-text/docs/basics#confidence-values
        # "Your code should not require the confidence field as it is not guaranteed to be accurate, or even set, in any of the results."
        confidence = best_hypothesis.get("confidence", 0.5)
        if with_confidence:
            return best_hypothesis["transcript"], confidence
        return best_hypothesis["transcript"]

    def recognize_google_cloud(self, audio_data, credentials_json=None, language="en-US", preferred_phrases=None, show_all=False):
        """
        Performs speech recognition on ``audio_data`` (an ``AudioData`` instance), using the Google Cloud Speech API.

        This function requires a Google Cloud Platform account; see the `Google Cloud Speech API Quickstart <https://cloud.google.com/speech/docs/getting-started>`__ for details and instructions. Basically, create a project, enable billing for the project, enable the Google Cloud Speech API for the project, and set up Service Account Key credentials for the project. The result is a JSON file containing the API credentials. The text content of this JSON file is specified by ``credentials_json``. If not specified, the library will try to automatically `find the default API credentials JSON file <https://developers.google.com/identity/protocols/application-default-credentials>`__.

        The recognition language is determined by ``language``, which is a BCP-47 language tag like ``"en-US"`` (US English). A list of supported language tags can be found in the `Google Cloud Speech API documentation <https://cloud.google.com/speech/docs/languages>`__.

        If ``preferred_phrases`` is an iterable of phrase strings, those given phrases will be more likely to be recognized over similar-sounding alternatives. This is useful for things like keyword/command recognition or adding new phrases that aren't in Google's vocabulary. Note that the API imposes certain `restrictions on the list of phrase strings <https://cloud.google.com/speech/limits#content>`__.

        Returns the most likely transcription if ``show_all`` is False (the default). Otherwise, returns the raw API response as a JSON dictionary.

        Raises a ``speech_recognition.UnknownValueError`` exception if the speech is unintelligible. Raises a ``speech_recognition.RequestError`` exception if the speech recognition operation failed, if the credentials aren't valid, or if there is no Internet connection.
        """
        assert isinstance(audio_data, AudioData), "``audio_data`` must be audio data"
        if credentials_json is None:
            assert os.environ.get('GOOGLE_APPLICATION_CREDENTIALS') is not None
        assert isinstance(language, str), "``language`` must be a string"
        assert preferred_phrases is None or all(isinstance(preferred_phrases, (type(""), type(u""))) for preferred_phrases in preferred_phrases), "``preferred_phrases`` must be a list of strings"

        try:
            import socket
            from google.cloud import speech
            from google.api_core.exceptions import GoogleAPICallError
        except ImportError:
            raise RequestError('missing google-cloud-speech module: ensure that google-cloud-speech is set up correctly.')

        if credentials_json is not None:
            client = speech.SpeechClient.from_service_account_json(credentials_json)
        else:
            client = speech.SpeechClient()

        flac_data = audio_data.get_flac_data(
            convert_rate=None if 8000 <= audio_data.sample_rate <= 48000 else max(8000, min(audio_data.sample_rate, 48000)),  # audio sample rate must be between 8 kHz and 48 kHz inclusive - clamp sample rate into this range
            convert_width=2  # audio samples must be 16-bit
        )
        audio = speech.RecognitionAudio(content=flac_data)

        config = {
            'encoding': speech.RecognitionConfig.AudioEncoding.FLAC,
            'sample_rate_hertz': audio_data.sample_rate,
            'language_code': language
        }
        if preferred_phrases is not None:
            config['speechContexts'] = [speech.SpeechContext(
                phrases=preferred_phrases
            )]
        if show_all:
            config['enableWordTimeOffsets'] = True  # some useful extra options for when we want all the output

        opts = {}
        if self.operation_timeout and socket.getdefaulttimeout() is None:
            opts['timeout'] = self.operation_timeout

        config = speech.RecognitionConfig(**config)

        try:
            response = client.recognize(config=config, audio=audio)
        except GoogleAPICallError as e:
            raise RequestError(e)
        except URLError as e:
            raise RequestError("recognition connection failed: {0}".format(e.reason))

        if show_all: return response
        if len(response.results) == 0: raise UnknownValueError()

        transcript = ''
        for result in response.results:
            transcript += result.alternatives[0].transcript.strip() + ' '
        return transcript

    def recognize_wit(self, audio_data, key, show_all=False):
        """
        Performs speech recognition on ``audio_data`` (an ``AudioData`` instance), using the Wit.ai API.

        The Wit.ai API key is specified by ``key``. Unfortunately, these are not available without `signing up for an account <https://wit.ai/>`__ and creating an app. You will need to add at least one intent to the app before you can see the API key, though the actual intent settings don't matter.

        To get the API key for a Wit.ai app, go to the app's overview page, go to the section titled "Make an API request", and look for something along the lines of ``Authorization: Bearer XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX``; ``XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX`` is the API key. Wit.ai API keys are 32-character uppercase alphanumeric strings.

        The recognition language is configured in the Wit.ai app settings.

        Returns the most likely transcription if ``show_all`` is false (the default). Otherwise, returns the `raw API response <https://wit.ai/docs/http/20141022#get-intent-via-text-link>`__ as a JSON dictionary.

        Raises a ``speech_recognition.UnknownValueError`` exception if the speech is unintelligible. Raises a ``speech_recognition.RequestError`` exception if the speech recognition operation failed, if the key isn't valid, or if there is no internet connection.
        """
        assert isinstance(audio_data, AudioData), "Data must be audio data"
        assert isinstance(key, str), "``key`` must be a string"

        wav_data = audio_data.get_wav_data(
            convert_rate=None if audio_data.sample_rate >= 8000 else 8000,  # audio samples must be at least 8 kHz
            convert_width=2  # audio samples should be 16-bit
        )
        url = "https://api.wit.ai/speech?v=20170307"
        request = Request(url, data=wav_data, headers={"Authorization": "Bearer {}".format(key), "Content-Type": "audio/wav"})
        try:
            response = urlopen(request, timeout=self.operation_timeout)
        except HTTPError as e:
            raise RequestError("recognition request failed: {}".format(e.reason))
        except URLError as e:
            raise RequestError("recognition connection failed: {}".format(e.reason))
        response_text = response.read().decode("utf-8")
        result = json.loads(response_text)

        # return results
        if show_all: return result
        if "_text" not in result or result["_text"] is None: raise UnknownValueError()
        return result["_text"]

    def recognize_azure(self, audio_data, key, language="en-US", profanity="masked", location="westus", show_all=False):
        """
        Performs speech recognition on ``audio_data`` (an ``AudioData`` instance), using the Microsoft Azure Speech API.

        The Microsoft Azure Speech API key is specified by ``key``. Unfortunately, these are not available without `signing up for an account <https://azure.microsoft.com/en-ca/pricing/details/cognitive-services/speech-api/>`__ with Microsoft Azure.

        To get the API key, go to the `Microsoft Azure Portal Resources <https://portal.azure.com/>`__ page, go to "All Resources" > "Add" > "See All" > Search "Speech > "Create", and fill in the form to make a "Speech" resource. On the resulting page (which is also accessible from the "All Resources" page in the Azure Portal), go to the "Show Access Keys" page, which will have two API keys, either of which can be used for the `key` parameter. Microsoft Azure Speech API keys are 32-character lowercase hexadecimal strings.

        The recognition language is determined by ``language``, a BCP-47 language tag like ``"en-US"`` (US English) or ``"fr-FR"`` (International French), defaulting to US English. A list of supported language values can be found in the `API documentation <https://docs.microsoft.com/en-us/azure/cognitive-services/speech/api-reference-rest/bingvoicerecognition#recognition-language>`__ under "Interactive and dictation mode".

        Returns the most likely transcription if ``show_all`` is false (the default). Otherwise, returns the `raw API response <https://docs.microsoft.com/en-us/azure/cognitive-services/speech/api-reference-rest/bingvoicerecognition#sample-responses>`__ as a JSON dictionary.

        Raises a ``speech_recognition.UnknownValueError`` exception if the speech is unintelligible. Raises a ``speech_recognition.RequestError`` exception if the speech recognition operation failed, if the key isn't valid, or if there is no internet connection.
        """
        assert isinstance(audio_data, AudioData), "Data must be audio data"
        assert isinstance(key, str), "``key`` must be a string"
        # assert isinstance(result_format, str), "``format`` must be a string" # simple|detailed
        assert isinstance(language, str), "``language`` must be a string"

        result_format = 'detailed'
        access_token, expire_time = getattr(self, "azure_cached_access_token", None), getattr(self, "azure_cached_access_token_expiry", None)
        allow_caching = True
        try:
            from time import monotonic  # we need monotonic time to avoid being affected by system clock changes, but this is only available in Python 3.3+
        except ImportError:
            expire_time = None  # monotonic time not available, don't cache access tokens
            allow_caching = False  # don't allow caching, since monotonic time isn't available
        if expire_time is None or monotonic() > expire_time:  # caching not enabled, first credential request, or the access token from the previous one expired
            # get an access token using OAuth
            credential_url = "https://" + location + ".api.cognitive.microsoft.com/sts/v1.0/issueToken"
            credential_request = Request(credential_url, data=b"", headers={
                "Content-type": "application/x-www-form-urlencoded",
                "Content-Length": "0",
                "Ocp-Apim-Subscription-Key": key,
            })

            if allow_caching:
                start_time = monotonic()

            try:
                credential_response = urlopen(credential_request, timeout=60)  # credential response can take longer, use longer timeout instead of default one
            except HTTPError as e:
                raise RequestError("credential request failed: {}".format(e.reason))
            except URLError as e:
                raise RequestError("credential connection failed: {}".format(e.reason))
            access_token = credential_response.read().decode("utf-8")

            if allow_caching:
                # save the token for the duration it is valid for
                self.azure_cached_access_token = access_token
                self.azure_cached_access_token_expiry = start_time + 600  # according to https://docs.microsoft.com/en-us/azure/cognitive-services/Speech-Service/rest-apis#authentication, the token expires in exactly 10 minutes

        wav_data = audio_data.get_wav_data(
            convert_rate=16000,  # audio samples must be 8kHz or 16 kHz
            convert_width=2  # audio samples should be 16-bit
        )

        url = "https://" + location + ".stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?{}".format(urlencode({
            "language": language,
            "format": result_format,
            "profanity": profanity
        }))

        if sys.version_info >= (3, 6):  # chunked-transfer requests are only supported in the standard library as of Python 3.6+, use it if possible
            request = Request(url, data=io.BytesIO(wav_data), headers={
                "Authorization": "Bearer {}".format(access_token),
                "Content-type": "audio/wav; codec=\"audio/pcm\"; samplerate=16000",
                "Transfer-Encoding": "chunked",
            })
        else:  # fall back on manually formatting the POST body as a chunked request
            ascii_hex_data_length = "{:X}".format(len(wav_data)).encode("utf-8")
            chunked_transfer_encoding_data = ascii_hex_data_length + b"\r\n" + wav_data + b"\r\n0\r\n\r\n"
            request = Request(url, data=chunked_transfer_encoding_data, headers={
                "Authorization": "Bearer {}".format(access_token),
                "Content-type": "audio/wav; codec=\"audio/pcm\"; samplerate=16000",
                "Transfer-Encoding": "chunked",
            })

        try:
            response = urlopen(request, timeout=self.operation_timeout)
        except HTTPError as e:
            raise RequestError("recognition request failed: {}".format(e.reason))
        except URLError as e:
            raise RequestError("recognition connection failed: {}".format(e.reason))
        response_text = response.read().decode("utf-8")
        result = json.loads(response_text)

        # return results
        if show_all:
            return result
        if "RecognitionStatus" not in result or result["RecognitionStatus"] != "Success" or "NBest" not in result:
            raise UnknownValueError()
        return result['NBest'][0]["Display"], result['NBest'][0]["Confidence"]

    def recognize_bing(self, audio_data, key, language="en-US", show_all=False):
        """
        Performs speech recognition on ``audio_data`` (an ``AudioData`` instance), using the Microsoft Bing Speech API.

        The Microsoft Bing Speech API key is specified by ``key``. Unfortunately, these are not available without `signing up for an account <https://azure.microsoft.com/en-ca/pricing/details/cognitive-services/speech-api/>`__ with Microsoft Azure.

        To get the API key, go to the `Microsoft Azure Portal Resources <https://portal.azure.com/>`__ page, go to "All Resources" > "Add" > "See All" > Search "Bing Speech API > "Create", and fill in the form to make a "Bing Speech API" resource. On the resulting page (which is also accessible from the "All Resources" page in the Azure Portal), go to the "Show Access Keys" page, which will have two API keys, either of which can be used for the `key` parameter. Microsoft Bing Speech API keys are 32-character lowercase hexadecimal strings.

        The recognition language is determined by ``language``, a BCP-47 language tag like ``"en-US"`` (US English) or ``"fr-FR"`` (International French), defaulting to US English. A list of supported language values can be found in the `API documentation <https://docs.microsoft.com/en-us/azure/cognitive-services/speech/api-reference-rest/bingvoicerecognition#recognition-language>`__ under "Interactive and dictation mode".

        Returns the most likely transcription if ``show_all`` is false (the default). Otherwise, returns the `raw API response <https://docs.microsoft.com/en-us/azure/cognitive-services/speech/api-reference-rest/bingvoicerecognition#sample-responses>`__ as a JSON dictionary.

        Raises a ``speech_recognition.UnknownValueError`` exception if the speech is unintelligible. Raises a ``speech_recognition.RequestError`` exception if the speech recognition operation failed, if the key isn't valid, or if there is no internet connection.
        """
        assert isinstance(audio_data, AudioData), "Data must be audio data"
        assert isinstance(key, str), "``key`` must be a string"
        assert isinstance(language, str), "``language`` must be a string"

        access_token, expire_time = getattr(self, "bing_cached_access_token", None), getattr(self, "bing_cached_access_token_expiry", None)
        allow_caching = True
        try:
            from time import monotonic  # we need monotonic time to avoid being affected by system clock changes, but this is only available in Python 3.3+
        except ImportError:
            expire_time = None  # monotonic time not available, don't cache access tokens
            allow_caching = False  # don't allow caching, since monotonic time isn't available
        if expire_time is None or monotonic() > expire_time:  # caching not enabled, first credential request, or the access token from the previous one expired
            # get an access token using OAuth
            credential_url = "https://api.cognitive.microsoft.com/sts/v1.0/issueToken"
            credential_request = Request(credential_url, data=b"", headers={
                "Content-type": "application/x-www-form-urlencoded",
                "Content-Length": "0",
                "Ocp-Apim-Subscription-Key": key,
            })

            if allow_caching:
                start_time = monotonic()

            try:
                credential_response = urlopen(credential_request, timeout=60)  # credential response can take longer, use longer timeout instead of default one
            except HTTPError as e:
                raise RequestError("credential request failed: {}".format(e.reason))
            except URLError as e:
                raise RequestError("credential connection failed: {}".format(e.reason))
            access_token = credential_response.read().decode("utf-8")

            if allow_caching:
                # save the token for the duration it is valid for
                self.bing_cached_access_token = access_token
                self.bing_cached_access_token_expiry = start_time + 600  # according to https://docs.microsoft.com/en-us/azure/cognitive-services/speech/api-reference-rest/bingvoicerecognition, the token expires in exactly 10 minutes

        wav_data = audio_data.get_wav_data(
            convert_rate=16000,  # audio samples must be 8kHz or 16 kHz
            convert_width=2  # audio samples should be 16-bit
        )

        url = "https://speech.platform.bing.com/speech/recognition/interactive/cognitiveservices/v1?{}".format(urlencode({
            "language": language,
            "locale": language,
            "requestid": uuid.uuid4(),
        }))

        if sys.version_info >= (3, 6):  # chunked-transfer requests are only supported in the standard library as of Python 3.6+, use it if possible
            request = Request(url, data=io.BytesIO(wav_data), headers={
                "Authorization": "Bearer {}".format(access_token),
                "Content-type": "audio/wav; codec=\"audio/pcm\"; samplerate=16000",
                "Transfer-Encoding": "chunked",
            })
        else:  # fall back on manually formatting the POST body as a chunked request
            ascii_hex_data_length = "{:X}".format(len(wav_data)).encode("utf-8")
            chunked_transfer_encoding_data = ascii_hex_data_length + b"\r\n" + wav_data + b"\r\n0\r\n\r\n"
            request = Request(url, data=chunked_transfer_encoding_data, headers={
                "Authorization": "Bearer {}".format(access_token),
                "Content-type": "audio/wav; codec=\"audio/pcm\"; samplerate=16000",
                "Transfer-Encoding": "chunked",
            })

        try:
            response = urlopen(request, timeout=self.operation_timeout)
        except HTTPError as e:
            raise RequestError("recognition request failed: {}".format(e.reason))
        except URLError as e:
            raise RequestError("recognition connection failed: {}".format(e.reason))
        response_text = response.read().decode("utf-8")
        result = json.loads(response_text)

        # return results
        if show_all: return result
        if "RecognitionStatus" not in result or result["RecognitionStatus"] != "Success" or "DisplayText" not in result: raise UnknownValueError()
        return result["DisplayText"]

    def recognize_lex(self, audio_data, bot_name, bot_alias, user_id, content_type="audio/l16; rate=16000; channels=1", access_key_id=None, secret_access_key=None, region=None):
        """
        Performs speech recognition on ``audio_data`` (an ``AudioData`` instance), using the Amazon Lex API.

        If access_key_id or secret_access_key is not set it will go through the list in the link below
        http://boto3.readthedocs.io/en/latest/guide/configuration.html#configuring-credentials
        """
        assert isinstance(audio_data, AudioData), "Data must be audio data"
        assert isinstance(bot_name, str), "``bot_name`` must be a string"
        assert isinstance(bot_alias, str), "``bot_alias`` must be a string"
        assert isinstance(user_id, str), "``user_id`` must be a string"
        assert isinstance(content_type, str), "``content_type`` must be a string"
        assert access_key_id is None or isinstance(access_key_id, str), "``access_key_id`` must be a string"
        assert secret_access_key is None or isinstance(secret_access_key, str), "``secret_access_key`` must be a string"
        assert region is None or isinstance(region, str), "``region`` must be a string"

        try:
            import boto3
        except ImportError:
            raise RequestError("missing boto3 module: ensure that boto3 is set up correctly.")

        client = boto3.client('lex-runtime', aws_access_key_id=access_key_id,
                              aws_secret_access_key=secret_access_key,
                              region_name=region)

        raw_data = audio_data.get_raw_data(
            convert_rate=16000, convert_width=2
        )

        accept = "text/plain; charset=utf-8"
        response = client.post_content(botName=bot_name, botAlias=bot_alias, userId=user_id, contentType=content_type, accept=accept, inputStream=raw_data)

        return response["inputTranscript"]

    def recognize_houndify(self, audio_data, client_id, client_key, show_all=False):
        """
        Performs speech recognition on ``audio_data`` (an ``AudioData`` instance), using the Houndify API.

        The Houndify client ID and client key are specified by ``client_id`` and ``client_key``, respectively. Unfortunately, these are not available without `signing up for an account <https://www.houndify.com/signup>`__. Once logged into the `dashboard <https://www.houndify.com/dashboard>`__, you will want to select "Register a new client", and fill in the form as necessary. When at the "Enable Domains" page, enable the "Speech To Text Only" domain, and then select "Save & Continue".

        To get the client ID and client key for a Houndify client, go to the `dashboard <https://www.houndify.com/dashboard>`__ and select the client's "View Details" link. On the resulting page, the client ID and client key will be visible. Client IDs and client keys are both Base64-encoded strings.

        Currently, only English is supported as a recognition language.

        Returns the most likely transcription if ``show_all`` is false (the default). Otherwise, returns the raw API response as a JSON dictionary.

        Raises a ``speech_recognition.UnknownValueError`` exception if the speech is unintelligible. Raises a ``speech_recognition.RequestError`` exception if the speech recognition operation failed, if the key isn't valid, or if there is no internet connection.
        """
        assert isinstance(audio_data, AudioData), "Data must be audio data"
        assert isinstance(client_id, str), "``client_id`` must be a string"
        assert isinstance(client_key, str), "``client_key`` must be a string"

        wav_data = audio_data.get_wav_data(
            convert_rate=None if audio_data.sample_rate in [8000, 16000] else 16000,  # audio samples must be 8 kHz or 16 kHz
            convert_width=2  # audio samples should be 16-bit
        )
        url = "https://api.houndify.com/v1/audio"
        user_id, request_id = str(uuid.uuid4()), str(uuid.uuid4())
        request_time = str(int(time.time()))
        request_signature = base64.urlsafe_b64encode(
            hmac.new(
                base64.urlsafe_b64decode(client_key),
                user_id.encode("utf-8") + b";" + request_id.encode("utf-8") + request_time.encode("utf-8"),
                hashlib.sha256
            ).digest()  # get the HMAC digest as bytes
        ).decode("utf-8")
        request = Request(url, data=wav_data, headers={
            "Content-Type": "application/json",
            "Hound-Request-Info": json.dumps({"ClientID": client_id, "UserID": user_id}),
            "Hound-Request-Authentication": "{};{}".format(user_id, request_id),
            "Hound-Client-Authentication": "{};{};{}".format(client_id, request_time, request_signature)
        })
        try:
            response = urlopen(request, timeout=self.operation_timeout)
        except HTTPError as e:
            raise RequestError("recognition request failed: {}".format(e.reason))
        except URLError as e:
            raise RequestError("recognition connection failed: {}".format(e.reason))
        response_text = response.read().decode("utf-8")
        result = json.loads(response_text)

        # return results
        if show_all: return result
        if "Disambiguation" not in result or result["Disambiguation"] is None:
            raise UnknownValueError()
        return result['Disambiguation']['ChoiceData'][0]['Transcription'], result['Disambiguation']['ChoiceData'][0]['ConfidenceScore']

    def recognize_amazon(self, audio_data, bucket_name=None, access_key_id=None, secret_access_key=None, region=None, job_name=None, file_key=None):
        """
        Performs speech recognition on ``audio_data`` (an ``AudioData`` instance) using Amazon Transcribe.
        https://aws.amazon.com/transcribe/
        If access_key_id or secret_access_key is not set it will go through the list in the link below
        http://boto3.readthedocs.io/en/latest/guide/configuration.html#configuring-credentials
        """
        assert access_key_id is None or isinstance(access_key_id, str), "``access_key_id`` must be a string"
        assert secret_access_key is None or isinstance(secret_access_key, str), "``secret_access_key`` must be a string"
        assert region is None or isinstance(region, str), "``region`` must be a string"
        import traceback
        import uuid
        import multiprocessing
        from botocore.exceptions import ClientError
        proc = multiprocessing.current_process()

        check_existing = audio_data is None and job_name

        bucket_name = bucket_name or ('%s-%s' % (str(uuid.uuid4()), proc.pid))
        job_name = job_name or ('%s-%s' % (str(uuid.uuid4()), proc.pid))

        try:
            import boto3
        except ImportError:
            raise RequestError("missing boto3 module: ensure that boto3 is set up correctly.")

        transcribe = boto3.client(
            'transcribe',
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name=region)

        s3 = boto3.client('s3', 
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name=region)

        session = boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name=region
        )

        # Upload audio data to S3.
        filename = '%s.wav' % job_name
        try:
            # Bucket creation fails surprisingly often, even if the bucket exists.
            # print('Attempting to create bucket %s...' % bucket_name)
            s3.create_bucket(Bucket=bucket_name)
        except ClientError as exc:
            print('Error creating bucket %s: %s' % (bucket_name, exc))
        s3res = session.resource('s3')
        bucket = s3res.Bucket(bucket_name)
        if audio_data is not None:
            print('Uploading audio data...')
            wav_data = audio_data.get_wav_data()
            s3.put_object(Bucket=bucket_name, Key=filename, Body=wav_data)
            object_acl = s3res.ObjectAcl(bucket_name, filename)
            object_acl.put(ACL='public-read')
        else:
            print('Skipping audio upload.')
        job_uri = 'https://%s.s3.amazonaws.com/%s' % (bucket_name, filename)

        if check_existing:

            # Wait for job to complete.
            try:
                status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
            except ClientError as exc:
                print('!'*80)
                print('Error getting job:', exc.response)
                if exc.response['Error']['Code'] == 'BadRequestException' and "The requested job couldn't be found" in str(exc):
                    # Some error caused the job we recorded to not exist on AWS.
                    # Likely we were interrupted right after retrieving and deleting the job but before recording the transcript.
                    # Reset and try again later.
                    exc = TranscriptionNotReady()
                    exc.job_name = None
                    exc.file_key = None
                    raise exc
                else:
                    # Some other error happened, so re-raise.
                    raise
            
            job = status['TranscriptionJob']
            if job['TranscriptionJobStatus'] in ['COMPLETED'] and 'TranscriptFileUri' in job['Transcript']:

                # Retrieve transcription JSON containing transcript.
                transcript_uri = job['Transcript']['TranscriptFileUri']
                import urllib.request, json
                with urllib.request.urlopen(transcript_uri) as json_data:
                    d = json.load(json_data)
                    confidences = []
                    for item in d['results']['items']:
                        confidences.append(float(item['alternatives'][0]['confidence']))
                    confidence = 0.5
                    if confidences:
                        confidence = sum(confidences)/float(len(confidences))
                    transcript = d['results']['transcripts'][0]['transcript']

                    # Delete job.
                    try:
                        transcribe.delete_transcription_job(TranscriptionJobName=job_name) # cleanup
                    except Exception as exc:
                        print('Warning, could not clean up transcription: %s' % exc)
                        traceback.print_exc()

                    # Delete S3 file.
                    s3.delete_object(Bucket=bucket_name, Key=filename)

                    return transcript, confidence
            elif job['TranscriptionJobStatus'] in ['FAILED']:
            
                # Delete job.
                try:
                    transcribe.delete_transcription_job(TranscriptionJobName=job_name) # cleanup
                except Exception as exc:
                    print('Warning, could not clean up transcription: %s' % exc)
                    traceback.print_exc()

                # Delete S3 file.
                s3.delete_object(Bucket=bucket_name, Key=filename)
                
                exc = TranscriptionFailed()
                exc.job_name = None
                exc.file_key = None
                raise exc
            else:
                # Keep waiting.
                print('Keep waiting.')
                exc = TranscriptionNotReady()
                exc.job_name = job_name
                exc.file_key = None
                raise exc

        else:

            # Launch the transcription job.
            # try:
                # transcribe.delete_transcription_job(TranscriptionJobName=job_name) # pre-cleanup
            # except:
                # # It's ok if this fails because the job hopefully doesn't exist yet.
                # pass
            try:
                transcribe.start_transcription_job(
                    TranscriptionJobName=job_name,
                    Media={'MediaFileUri': job_uri},
                    MediaFormat='wav',
                    LanguageCode='en-US'
                )
                exc = TranscriptionNotReady()
                exc.job_name = job_name
                exc.file_key = None
                raise exc
            except ClientError as exc:
                print('!'*80)
                print('Error starting job:', exc.response)
                if exc.response['Error']['Code'] == 'LimitExceededException':
                    # Could not start job. Cancel everything.
                    s3.delete_object(Bucket=bucket_name, Key=filename)
                    exc = TranscriptionNotReady()
                    exc.job_name = None
                    exc.file_key = None
                    raise exc
                else:
                    # Some other error happened, so re-raise.
                    raise

    def recognize_assemblyai(self, audio_data, api_token, job_name=None, **kwargs):
        """
        Wraps the AssemblyAI STT service.
        https://www.assemblyai.com/
        """

        def read_file(filename, chunk_size=5242880):
            with open(filename, 'rb') as _file:
                while True:
                    data = _file.read(chunk_size)
                    if not data:
                        break
                    yield data

        check_existing = audio_data is None and job_name
        if check_existing:
            # Query status.
            transciption_id = job_name
            endpoint = f"https://api.assemblyai.com/v2/transcript/{transciption_id}"
            headers = {
                "authorization": api_token,
            }
            response = requests.get(endpoint, headers=headers)
            data = response.json()
            status = data['status']

            if status == 'error':
                # Handle error.
                exc = TranscriptionFailed()
                exc.job_name = None
                exc.file_key = None
                raise exc
                # Handle success.
            elif status == 'completed':
                confidence = data['confidence']
                text = data['text']
                return text, confidence

            # Otherwise keep waiting.
            print('Keep waiting.')
            exc = TranscriptionNotReady()
            exc.job_name = job_name
            exc.file_key = None
            raise exc
        else:
            # Upload file.
            headers = {'authorization': api_token}
            response = requests.post('https://api.assemblyai.com/v2/upload',
                                     headers=headers,
                                     data=read_file(audio_data))
            upload_url = response.json()['upload_url']

            # Queue file for transcription.
            endpoint = "https://api.assemblyai.com/v2/transcript"
            json = {
              "audio_url": upload_url
            }
            headers = {
                "authorization": api_token,
                "content-type": "application/json"
            }
            response = requests.post(endpoint, json=json, headers=headers)
            data = response.json()
            transciption_id = data['id']
            exc = TranscriptionNotReady()
            exc.job_name = transciption_id
            exc.file_key = None
            raise exc

    def recognize_ibm(self, audio_data, key, language="en-US", show_all=False):
        """
        Performs speech recognition on ``audio_data`` (an ``AudioData`` instance), using the IBM Speech to Text API.

        The IBM Speech to Text username and password are specified by ``username`` and ``password``, respectively. Unfortunately, these are not available without `signing up for an account <https://console.ng.bluemix.net/registration/>`__. Once logged into the Bluemix console, follow the instructions for `creating an IBM Watson service instance <https://www.ibm.com/watson/developercloud/doc/getting_started/gs-credentials.shtml>`__, where the Watson service is "Speech To Text". IBM Speech to Text usernames are strings of the form XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX, while passwords are mixed-case alphanumeric strings.

        The recognition language is determined by ``language``, an RFC5646 language tag with a dialect like ``"en-US"`` (US English) or ``"zh-CN"`` (Mandarin Chinese), defaulting to US English. The supported language values are listed under the ``model`` parameter of the `audio recognition API documentation <https://www.ibm.com/watson/developercloud/speech-to-text/api/v1/#sessionless_methods>`__, in the form ``LANGUAGE_BroadbandModel``, where ``LANGUAGE`` is the language value.

        Returns the most likely transcription if ``show_all`` is false (the default). Otherwise, returns the `raw API response <https://www.ibm.com/watson/developercloud/speech-to-text/api/v1/#sessionless_methods>`__ as a JSON dictionary.

        Raises a ``speech_recognition.UnknownValueError`` exception if the speech is unintelligible. Raises a ``speech_recognition.RequestError`` exception if the speech recognition operation failed, if the key isn't valid, or if there is no internet connection.
        """
        assert isinstance(audio_data, AudioData), "Data must be audio data"
        assert isinstance(key, str), "``key`` must be a string"

        flac_data = audio_data.get_flac_data(
            convert_rate=None if audio_data.sample_rate >= 16000 else 16000,  # audio samples should be at least 16 kHz
            convert_width=None if audio_data.sample_width >= 2 else 2  # audio samples should be at least 16-bit
        )
        url = "https://gateway-wdc.watsonplatform.net/speech-to-text/api/v1/recognize"
        request = Request(url, data=flac_data, headers={
            "Content-Type": "audio/x-flac",
        })
        request.get_method = lambda: 'POST'
        username = 'apikey'
        password = key
        authorization_value = base64.standard_b64encode("{}:{}".format(username, password).encode("utf-8")).decode("utf-8")
        request.add_header("Authorization", "Basic {}".format(authorization_value))
        try:
            response = urlopen(request, timeout=self.operation_timeout)
        except HTTPError as e:
            raise RequestError("recognition request failed: {}".format(e.reason))
        except URLError as e:
            raise RequestError("recognition connection failed: {}".format(e.reason))
        response_text = response.read().decode("utf-8")
        result = json.loads(response_text)

        # return results
        if show_all:
            return result
        if "results" not in result or len(result["results"]) < 1 or "alternatives" not in result["results"][0]:
            raise UnknownValueError()

        transcription = []
        confidence = None
        for utterance in result["results"]:
            if "alternatives" not in utterance: raise UnknownValueError()
            for hypothesis in utterance["alternatives"]:
                if "transcript" in hypothesis:
                    transcription.append(hypothesis["transcript"])
                    confidence = hypothesis["confidence"]
                    break
        return "\n".join(transcription), confidence

    lasttfgraph = ''
    tflabels = None

    def recognize_tensorflow(self, audio_data, tensor_graph='tensorflow-data/conv_actions_frozen.pb', tensor_label='tensorflow-data/conv_actions_labels.txt'):
        """
        Performs speech recognition on ``audio_data`` (an ``AudioData`` instance).

        Path to Tensor loaded from ``tensor_graph``. You can download a model here: http://download.tensorflow.org/models/speech_commands_v0.01.zip

        Path to Tensor Labels file loaded from ``tensor_label``.
        """
        assert isinstance(audio_data, AudioData), "Data must be audio data"
        assert isinstance(tensor_graph, str), "``tensor_graph`` must be a string"
        assert isinstance(tensor_label, str), "``tensor_label`` must be a string"

        try:
            import tensorflow as tf
        except ImportError:
            raise RequestError("missing tensorflow module: ensure that tensorflow is set up correctly.")

        if not (tensor_graph == self.lasttfgraph):
            self.lasttfgraph = tensor_graph

            # load graph
            with tf.gfile.FastGFile(tensor_graph, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
            # load labels
            self.tflabels = [line.rstrip() for line in tf.gfile.GFile(tensor_label)]

        wav_data = audio_data.get_wav_data(
            convert_rate=16000, convert_width=2
        )

        with tf.Session() as sess:
            input_layer_name = 'wav_data:0'
            output_layer_name = 'labels_softmax:0'
            softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
            predictions, = sess.run(softmax_tensor, {input_layer_name: wav_data})

            # Sort labels in order of confidence
            top_k = predictions.argsort()[-1:][::-1]
            for node_id in top_k:
                human_string = self.tflabels[node_id]
                return human_string

    def recognize_whisper(self, audio_data, model="base", show_dict=False, load_options=None, language=None, translate=False, **transcribe_options):
        """
        Performs speech recognition on ``audio_data`` (an ``AudioData`` instance), using Whisper.

        The recognition language is determined by ``language``, an uncapitalized full language name like "english" or "chinese". See the full language list at https://github.com/openai/whisper/blob/main/whisper/tokenizer.py

        model can be any of tiny, base, small, medium, large, tiny.en, base.en, small.en, medium.en. See https://github.com/openai/whisper for more details.

        If show_dict is true, returns the full dict response from Whisper, including the detected language. Otherwise returns only the transcription.

        You can translate the result to english with Whisper by passing translate=True

        Other values are passed directly to whisper. See https://github.com/openai/whisper/blob/main/whisper/transcribe.py for all options
        """

        assert isinstance(audio_data, AudioData), "Data must be audio data"
        import numpy as np
        import soundfile as sf
        import torch
        import whisper

        if load_options or not hasattr(self, "whisper_model") or self.whisper_model.get(model) is None:
            self.whisper_model = getattr(self, "whisper_model", {})
            self.whisper_model[model] = whisper.load_model(model, **load_options or {})

        # 16 kHz https://github.com/openai/whisper/blob/28769fcfe50755a817ab922a7bc83483159600a9/whisper/audio.py#L98-L99
        wav_bytes = audio_data.get_wav_data(convert_rate=16000)
        wav_stream = io.BytesIO(wav_bytes)
        audio_array, sampling_rate = sf.read(wav_stream)
        audio_array = audio_array.astype(np.float32)

        result = self.whisper_model[model].transcribe(
            audio_array,
            language=language,
            task="translate" if translate else None,
            fp16=torch.cuda.is_available(),
            **transcribe_options
        )

        if show_dict:
            return result
        else:
            return result["text"]

    recognize_whisper_api = whisper.recognize_whisper_api
            
    def recognize_vosk(self, audio_data, language='en'):
        from vosk import Model, KaldiRecognizer
        
        assert isinstance(audio_data, AudioData), "Data must be audio data"
        
        if not hasattr(self, 'vosk_model'):
            if not os.path.exists("model"):
                return "Please download the model from https://github.com/alphacep/vosk-api/blob/master/doc/models.md and unpack as 'model' in the current folder."
                exit (1)
            self.vosk_model = Model("model")

        rec = KaldiRecognizer(self.vosk_model, 16000);
        
        rec.AcceptWaveform(audio_data.get_raw_data(convert_rate=16000, convert_width=2));
        finalRecognition = rec.FinalResult()
        
        return finalRecognition


class PortableNamedTemporaryFile(object):
    """Limited replacement for ``tempfile.NamedTemporaryFile``, except unlike ``tempfile.NamedTemporaryFile``, the file can be opened again while it's currently open, even on Windows."""
    def __init__(self, mode="w+b"):
        self.mode = mode

    def __enter__(self):
        # create the temporary file and open it
        file_descriptor, file_path = tempfile.mkstemp()
        self._file = os.fdopen(file_descriptor, self.mode)

        # the name property is a public field
        self.name = file_path
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._file.close()
        os.remove(self.name)

    def write(self, *args, **kwargs):
        return self._file.write(*args, **kwargs)

    def writelines(self, *args, **kwargs):
        return self._file.writelines(*args, **kwargs)

    def flush(self, *args, **kwargs):
        return self._file.flush(*args, **kwargs)


# ===============================
#  backwards compatibility shims
# ===============================

WavFile = AudioFile  # WavFile was renamed to AudioFile in 3.4.1


def recognize_api(self, audio_data, client_access_token, language="en", session_id=None, show_all=False):
    wav_data = audio_data.get_wav_data(convert_rate=16000, convert_width=2)
    url = "https://api.api.ai/v1/query"
    while True:
        boundary = uuid.uuid4().hex
        if boundary.encode("utf-8") not in wav_data: break
    if session_id is None: session_id = uuid.uuid4().hex
    data = b"--" + boundary.encode("utf-8") + b"\r\n" + b"Content-Disposition: form-data; name=\"request\"\r\n" + b"Content-Type: application/json\r\n" + b"\r\n" + b"{\"v\": \"20150910\", \"sessionId\": \"" + session_id.encode("utf-8") + b"\", \"lang\": \"" + language.encode("utf-8") + b"\"}\r\n" + b"--" + boundary.encode("utf-8") + b"\r\n" + b"Content-Disposition: form-data; name=\"voiceData\"; filename=\"audio.wav\"\r\n" + b"Content-Type: audio/wav\r\n" + b"\r\n" + wav_data + b"\r\n" + b"--" + boundary.encode("utf-8") + b"--\r\n"
    request = Request(url, data=data, headers={"Authorization": "Bearer {}".format(client_access_token), "Content-Length": str(len(data)), "Expect": "100-continue", "Content-Type": "multipart/form-data; boundary={}".format(boundary)})
    try: response = urlopen(request, timeout=10)
    except HTTPError as e: raise RequestError("recognition request failed: {}".format(e.reason))
    except URLError as e: raise RequestError("recognition connection failed: {}".format(e.reason))
    response_text = response.read().decode("utf-8")
    result = json.loads(response_text)
    if show_all: return result
    if "status" not in result or "errorType" not in result["status"] or result["status"]["errorType"] != "success":
        raise UnknownValueError()
    return result["result"]["resolvedQuery"]


Recognizer.recognize_api = classmethod(recognize_api)  # API.AI Speech Recognition is deprecated/not recommended as of 3.5.0, and currently is only optionally available for paid plans
