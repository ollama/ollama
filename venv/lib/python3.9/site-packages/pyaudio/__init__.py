# PyAudio : Python Bindings for PortAudio.
#
# Copyright (c) 2006 Hubert Pham
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
PyAudio provides Python bindings for PortAudio, the cross-platform
audio I/O library. With PyAudio, you can easily use Python to play and
record audio on a variety of platforms.

.. include:: ../sphinx/examples.rst

Overview
--------

**Classes**
  :py:class:`PyAudio`, :py:class:`PyAudio.Stream`

.. only:: pamac

   **Host Specific Classes**
     :py:class:`PaMacCoreStreamInfo`

**Stream Conversion Convenience Functions**
  :py:func:`get_sample_size`, :py:func:`get_format_from_width`

**PortAudio version**
  :py:func:`get_portaudio_version`, :py:func:`get_portaudio_version_text`

.. |PaSampleFormat| replace:: :ref:`PortAudio Sample Format <PaSampleFormat>`
.. _PaSampleFormat:

**Portaudio Sample Formats**
  :py:data:`paFloat32`, :py:data:`paInt32`, :py:data:`paInt24`,
  :py:data:`paInt16`, :py:data:`paInt8`, :py:data:`paUInt8`,
  :py:data:`paCustomFormat`

.. |PaHostAPI| replace:: :ref:`PortAudio Host API <PaHostAPI>`
.. _PaHostAPI:

**PortAudio Host APIs**
  :py:data:`paInDevelopment`, :py:data:`paDirectSound`, :py:data:`paMME`,
  :py:data:`paASIO`, :py:data:`paSoundManager`, :py:data:`paCoreAudio`,
  :py:data:`paOSS`, :py:data:`paALSA`, :py:data:`paAL`, :py:data:`paBeOS`,
  :py:data:`paWDMKS`, :py:data:`paJACK`, :py:data:`paWASAPI`,
  :py:data:`paNoDevice`

.. |PaErrorCode| replace:: :ref:`PortAudio Error Code <PaErrorCode>`
.. _PaErrorCode:

**PortAudio Error Codes**
  :py:data:`paNoError`, :py:data:`paNotInitialized`,
  :py:data:`paUnanticipatedHostError`, :py:data:`paInvalidChannelCount`,
  :py:data:`paInvalidSampleRate`, :py:data:`paInvalidDevice`,
  :py:data:`paInvalidFlag`, :py:data:`paSampleFormatNotSupported`,
  :py:data:`paBadIODeviceCombination`, :py:data:`paInsufficientMemory`,
  :py:data:`paBufferTooBig`, :py:data:`paBufferTooSmall`,
  :py:data:`paNullCallback`, :py:data:`paBadStreamPtr`,
  :py:data:`paTimedOut`, :py:data:`paInternalError`,
  :py:data:`paDeviceUnavailable`,
  :py:data:`paIncompatibleHostApiSpecificStreamInfo`,
  :py:data:`paStreamIsStopped`, :py:data:`paStreamIsNotStopped`,
  :py:data:`paInputOverflowed`, :py:data:`paOutputUnderflowed`,
  :py:data:`paHostApiNotFound`, :py:data:`paInvalidHostApi`,
  :py:data:`paCanNotReadFromACallbackStream`,
  :py:data:`paCanNotWriteToACallbackStream`,
  :py:data:`paCanNotReadFromAnOutputOnlyStream`,
  :py:data:`paCanNotWriteToAnInputOnlyStream`,
  :py:data:`paIncompatibleStreamHostApi`

.. |PaCallbackReturnCodes| replace:: :ref:`PortAudio Callback Return Code <PaCallbackReturnCodes>`
.. _PaCallbackReturnCodes:

**PortAudio Callback Return Codes**
  :py:data:`paContinue`, :py:data:`paComplete`, :py:data:`paAbort`

.. |PaCallbackFlags| replace:: :ref:`PortAutio Callback Flag <PaCallbackFlags>`
.. _PaCallbackFlags:

**PortAudio Callback Flags**
  :py:data:`paInputUnderflow`, :py:data:`paInputOverflow`,
  :py:data:`paOutputUnderflow`, :py:data:`paOutputOverflow`,
  :py:data:`paPrimingOutput`
"""

__author__ = "Hubert Pham"
__version__ = "0.2.14"
__docformat__ = "restructuredtext en"

import locale
import warnings

try:
    import pyaudio._portaudio as pa
except ImportError:
    print("Could not import the PyAudio C module 'pyaudio._portaudio'.")
    raise


# PaSampleFormat Sample Formats

paFloat32 = pa.paFloat32  #: 32 bit float
paInt32 = pa.paInt32  #: 32 bit int
paInt24 = pa.paInt24  #: 24 bit int
paInt16 = pa.paInt16  #: 16 bit int
paInt8 = pa.paInt8  #: 8 bit int
paUInt8 = pa.paUInt8  #: 8 bit unsigned int
paCustomFormat = pa.paCustomFormat  #: a custom data format

# HostAPI TypeId

paInDevelopment = pa.paInDevelopment  #: Still in development
paDirectSound = pa.paDirectSound  #: DirectSound (Windows only)
paMME = pa.paMME  #: Multimedia Extension (Windows only)
paASIO = pa.paASIO  #: Steinberg Audio Stream Input/Output
paSoundManager = pa.paSoundManager  #: SoundManager (OSX only)
paCoreAudio = pa.paCoreAudio  #: CoreAudio (OSX only)
paOSS = pa.paOSS  #: Open Sound System (Linux only)
paALSA = pa.paALSA  #: Advanced Linux Sound Architecture (Linux only)
paAL = pa.paAL  #: Open Audio Library
paBeOS = pa.paBeOS  #: BeOS Sound System
paWDMKS = pa.paWDMKS  #: Windows Driver Model (Windows only)
paJACK = pa.paJACK  #: JACK Audio Connection Kit
paWASAPI = pa.paWASAPI  #: Windows Vista Audio stack architecture
paNoDevice = pa.paNoDevice  #: Not actually an audio device

# PortAudio Error Codes

paNoError = pa.paNoError
paNotInitialized = pa.paNotInitialized
paUnanticipatedHostError = pa.paUnanticipatedHostError
paInvalidChannelCount = pa.paInvalidChannelCount
paInvalidSampleRate = pa.paInvalidSampleRate
paInvalidDevice = pa.paInvalidDevice
paInvalidFlag = pa.paInvalidFlag
paSampleFormatNotSupported = pa.paSampleFormatNotSupported
paBadIODeviceCombination = pa.paBadIODeviceCombination
paInsufficientMemory = pa.paInsufficientMemory
paBufferTooBig = pa.paBufferTooBig
paBufferTooSmall = pa.paBufferTooSmall
paNullCallback = pa.paNullCallback
paBadStreamPtr = pa.paBadStreamPtr
paTimedOut = pa.paTimedOut
paInternalError = pa.paInternalError
paDeviceUnavailable = pa.paDeviceUnavailable
paIncompatibleHostApiSpecificStreamInfo = (
    pa.paIncompatibleHostApiSpecificStreamInfo)
paStreamIsStopped = pa.paStreamIsStopped
paStreamIsNotStopped = pa.paStreamIsNotStopped
paInputOverflowed = pa.paInputOverflowed
paOutputUnderflowed = pa.paOutputUnderflowed
paHostApiNotFound = pa.paHostApiNotFound
paInvalidHostApi = pa.paInvalidHostApi
paCanNotReadFromACallbackStream = pa.paCanNotReadFromACallbackStream
paCanNotWriteToACallbackStream = pa.paCanNotWriteToACallbackStream
paCanNotReadFromAnOutputOnlyStream = pa.paCanNotReadFromAnOutputOnlyStream
paCanNotWriteToAnInputOnlyStream = pa.paCanNotWriteToAnInputOnlyStream
paIncompatibleStreamHostApi = pa.paIncompatibleStreamHostApi

# PortAudio Callback Return Codes

paContinue = pa.paContinue  #: There is more audio data to come
paComplete = pa.paComplete  #: This was the last block of audio data
paAbort = pa.paAbort  #: An error ocurred, stop playback/recording

# PortAudio Callback Flags

paInputUnderflow = pa.paInputUnderflow  #: Buffer underflow in input
paInputOverflow = pa.paInputOverflow  #: Buffer overflow in input
paOutputUnderflow = pa.paOutputUnderflow  #: Buffer underflow in output
paOutputOverflow = pa.paOutputOverflow  #: Buffer overflow in output
paPrimingOutput = pa.paPrimingOutput  #: Just priming, not playing yet

# PortAudio Misc Constants

paFramesPerBufferUnspecified = pa.paFramesPerBufferUnspecified


# Utilities

def get_sample_size(format):
    """Returns the size (in bytes) for the specified sample *format*.

    :param format: A |PaSampleFormat| constant.
    :raises ValueError: on invalid specified `format`.
    :rtype: integer
    """
    return pa.get_sample_size(format)


def get_format_from_width(width, unsigned=True):
    """Returns a PortAudio format constant for the specified *width*.

    :param width: The desired sample width in bytes (1, 2, 3, or 4)
    :param unsigned: For 1 byte width, specifies signed or unsigned format.

    :raises ValueError: when invalid *width*
    :rtype: A |PaSampleFormat| constant
    """
    if width == 1:
        if unsigned:
            return paUInt8
        return paInt8
    if width == 2:
        return paInt16
    if width == 3:
        return paInt24
    if width == 4:
        return paFloat32

    raise ValueError(f"Invalid width: {width}")


# Versioning

def get_portaudio_version():
    """Returns portaudio version.

    :rtype: int
    """
    return pa.get_version()


def get_portaudio_version_text():
    """Returns PortAudio version as a text string.

    :rtype: string
    """
    return pa.get_version_text()


class PyAudio:
    """Python interface to PortAudio.

    Provides methods to:
     - initialize and terminate PortAudio
     - open and close streams
     - query and inspect the available PortAudio Host APIs
     - query and inspect the available PortAudio audio devices.

    **Stream Management**
      :py:func:`open`, :py:func:`close`

    **Host API**
      :py:func:`get_host_api_count`, :py:func:`get_default_host_api_info`,
      :py:func:`get_host_api_info_by_type`,
      :py:func:`get_host_api_info_by_index`,
      :py:func:`get_device_info_by_host_api_device_index`

    **Device API**
      :py:func:`get_device_count`, :py:func:`is_format_supported`,
      :py:func:`get_default_input_device_info`,
      :py:func:`get_default_output_device_info`,
      :py:func:`get_device_info_by_index`

    **Stream Format Conversion**
      :py:func:`get_sample_size`, :py:func:`get_format_from_width`

    **Details**
    """

    class Stream:
        """PortAudio Stream Wrapper. Use :py:func:`PyAudio.open` to instantiate.

        **Opening and Closing**
          :py:func:`__init__`, :py:func:`close`

        **Stream Info**
          :py:func:`get_input_latency`, :py:func:`get_output_latency`,
          :py:func:`get_time`, :py:func:`get_cpu_load`

        **Stream Management**
          :py:func:`start_stream`, :py:func:`stop_stream`, :py:func:`is_active`,
          :py:func:`is_stopped`

        **Input Output**
          :py:func:`write`, :py:func:`read`, :py:func:`get_read_available`,
          :py:func:`get_write_available`
        """
        def __init__(self,
                     PA_manager,
                     rate,
                     channels,
                     format,
                     input=False,
                     output=False,
                     input_device_index=None,
                     output_device_index=None,
                     frames_per_buffer=pa.paFramesPerBufferUnspecified,
                     start=True,
                     input_host_api_specific_stream_info=None,
                     output_host_api_specific_stream_info=None,
                     stream_callback=None):
            """Initialize an audio stream.

            Do not call directly. Use :py:func:`PyAudio.open`.

            A stream can either be input, output, or both.

            :param PA_manager: A reference to the managing :py:class:`PyAudio`
                instance
            :param rate: Sampling rate
            :param channels: Number of channels
            :param format: Sampling size and format. See |PaSampleFormat|.
            :param input: Specifies whether this is an input stream.
                Defaults to ``False``.
            :param output: Specifies whether this is an output stream.
                Defaults to ``False``.
            :param input_device_index: Index of Input Device to use.
                Unspecified (or ``None``) uses default device.
                Ignored if `input` is ``False``.
            :param output_device_index:
                Index of Output Device to use.
                Unspecified (or ``None``) uses the default device.
                Ignored if `output` is ``False``.
            :param frames_per_buffer: Specifies the number of frames per buffer.
            :param start: Start the stream running immediately.
                Defaults to ``True``. In general, there is no reason to set
                this to ``False``.
            :param input_host_api_specific_stream_info: Specifies a host API
                specific stream information data structure for input.

                .. only:: pamac

                   See :py:class:`PaMacCoreStreamInfo`.

            :param output_host_api_specific_stream_info: Specifies a host API
                specific stream information data structure for output.

                .. only:: pamac

                   See :py:class:`PaMacCoreStreamInfo`.

            :param stream_callback: Specifies a callback function for
                *non-blocking* (callback) operation.  Default is
                ``None``, which indicates *blocking* operation (i.e.,
                :py:func:`PyAudio.Stream.read` and
                :py:func:`PyAudio.Stream.write`).  To use non-blocking
                operation, specify a callback that conforms to the following
                signature:

                .. code-block:: python

                   callback(in_data,      # input data if input=True; else None
                            frame_count,  # number of frames
                            time_info,    # dictionary
                            status_flags) # PaCallbackFlags

                ``time_info`` is a dictionary with the following keys:
                ``input_buffer_adc_time``, ``current_time``, and
                ``output_buffer_dac_time``; see the PortAudio
                documentation for their meanings.  ``status_flags`` is one
                of |PaCallbackFlags|.

                The callback must return a tuple:

                .. code-block:: python

                    (out_data, flag)

                ``out_data`` is a byte array whose length should be the
                (``frame_count * channels * bytes-per-channel``) if
                ``output=True`` or ``None`` if ``output=False``.  ``flag``
                must be either :py:data:`paContinue`, :py:data:`paComplete` or
                :py:data:`paAbort` (one of |PaCallbackReturnCodes|).
                When ``output=True`` and ``out_data`` does not contain at
                least ``frame_count`` frames, :py:data:`paComplete` is
                assumed for ``flag``.

                **Note:** ``stream_callback`` is called in a separate
                thread (from the main thread).  Exceptions that occur in
                the ``stream_callback`` will:

                1. print a traceback on standard error to aid debugging,
                2. queue the exception to be thrown (at some point) in
                   the main thread, and
                3. return `paAbort` to PortAudio to stop the stream.

                **Note:** Do not call :py:func:`PyAudio.Stream.read` or
                :py:func:`PyAudio.Stream.write` if using non-blocking operation.

                **See:** PortAudio's callback signature for additional
                details: http://portaudio.com/docs/v19-doxydocs/portaudio_8h.html#a8a60fb2a5ec9cbade3f54a9c978e2710

            :raise ValueError: Neither input nor output are set True.
            """
            if not (input or output):
                raise ValueError("Must specify an input or output " + "stream.")

            self._parent = PA_manager
            self._is_input = input
            self._is_output = output
            self._is_running = start
            self._rate = rate
            self._channels = channels
            self._format = format
            self._frames_per_buffer = frames_per_buffer

            arguments = {
                'rate': rate,
                'channels': channels,
                'format': format,
                'input': input,
                'output': output,
                'input_device_index': input_device_index,
                'output_device_index': output_device_index,
                'frames_per_buffer': frames_per_buffer
            }

            if input_host_api_specific_stream_info:
                arguments[
                    'input_host_api_specific_stream_info'
                ] = input_host_api_specific_stream_info

            if output_host_api_specific_stream_info:
                arguments[
                    'output_host_api_specific_stream_info'
                ] = output_host_api_specific_stream_info

            if stream_callback:
                arguments['stream_callback'] = stream_callback

            # calling pa.open returns a stream object
            self._stream = pa.open(**arguments)

            self._input_latency = self._stream.inputLatency
            self._output_latency = self._stream.outputLatency

            if self._is_running:
                pa.start_stream(self._stream)

        def close(self):
            """Closes the stream."""
            pa.close(self._stream)
            self._is_running = False
            self._parent._remove_stream(self)

        # Stream Info

        def get_input_latency(self):
            """Returns the input latency.

            :rtype: float
            """
            return self._stream.inputLatency

        def get_output_latency(self):
            """Returns the output latency.

            :rtype: float
            """
            return self._stream.outputLatency

        def get_time(self):
            """Returns stream time.

            :rtype: float
            """
            return pa.get_stream_time(self._stream)

        def get_cpu_load(self):
            """Return the CPU load. Always 0.0 when using the blocking API.

            :rtype: float
            """
            return pa.get_stream_cpu_load(self._stream)

        # Stream Lifecycle

        def start_stream(self):
            """Starts the stream."""
            if self._is_running:
                return

            pa.start_stream(self._stream)
            self._is_running = True

        def stop_stream(self):
            """Stops the stream."""
            if not self._is_running:
                return

            pa.stop_stream(self._stream)
            self._is_running = False

        def is_active(self):
            """Returns whether the stream is active.

            :rtype: bool
            """
            return pa.is_stream_active(self._stream)

        def is_stopped(self):
            """Returns whether the stream is stopped.

            :rtype: bool
            """
            return pa.is_stream_stopped(self._stream)

        # Stream blocking I/O

        def write(self, frames, num_frames=None, exception_on_underflow=False):
            """Write samples to the stream for playback.

            Do not call when using non-blocking mode.

            :param frames:
               The frames of data.
            :param num_frames:
               The number of frames to write.
               Defaults to None, in which this value will be
               automatically computed.
            :param exception_on_underflow:
               Specifies whether an IOError exception should be thrown
               (or silently ignored) on buffer underflow. Defaults
               to False for improved performance, especially on
               slower platforms.

            :raises IOError: if the stream is not an output stream
               or if the write operation was unsuccessful.

            :rtype: `None`
            """
            if not self._is_output:
                raise IOError("Not output stream",
                              paCanNotWriteToAnInputOnlyStream)

            if num_frames is None:
                # Determine how many frames to read:
                width = get_sample_size(self._format)
                num_frames = int(len(frames) / (self._channels * width))

            pa.write_stream(self._stream, frames, num_frames,
                            exception_on_underflow)

        def read(self, num_frames, exception_on_overflow=True):
            """Read samples from the stream.

            Do not call when using non-blocking mode.

            :param num_frames: The number of frames to read.
            :param exception_on_overflow:
               Specifies whether an IOError exception should be thrown
               (or silently ignored) on input buffer overflow. Defaults
               to True.
            :raises IOError: if stream is not an input stream
              or if the read operation was unsuccessful.
            :rtype: bytes
            """
            if not self._is_input:
                raise IOError("Not input stream",
                              paCanNotReadFromAnOutputOnlyStream)
            return pa.read_stream(self._stream, num_frames,
                                  exception_on_overflow)

        def get_read_available(self):
            """Return the number of frames that can be read without waiting.

            :rtype: integer
            """
            return pa.get_stream_read_available(self._stream)

        def get_write_available(self):
            """Return the number of frames that can be written without waiting.

            :rtype: integer
            """
            return pa.get_stream_write_available(self._stream)

    # Initialization and Termination

    def __init__(self):
        """Initialize PortAudio."""
        pa.initialize()
        self._streams = set()

    def terminate(self):
        """Terminates PortAudio.

        :attention: Be sure to call this method for every instance of this
          object to release PortAudio resources.
        """
        for stream in self._streams.copy():
            stream.close()

        self._streams = set()
        pa.terminate()

    # Utilities

    def get_sample_size(self, format):
        """Returns the size (in bytes) for the specified sample `format`
        (a |PaSampleFormat| constant).

        :param format: A |PaSampleFormat| constant.
        :raises ValueError: Invalid specified `format`.
        :rtype: integer
        """
        return pa.get_sample_size(format)

    def get_format_from_width(self, width, unsigned=True):
        """Returns a PortAudio format constant for the specified `width`.

        :param width: The desired sample width in bytes (1, 2, 3, or 4)
        :param unsigned: For 1 byte width, specifies signed or unsigned format.

        :raises ValueError: for invalid `width`
        :rtype: A |PaSampleFormat| constant.
        """
        return get_format_from_width(width, unsigned)

    # Stream Factory

    def open(self, *args, **kwargs):
        """Opens a new stream.

        See constructor for :py:func:`PyAudio.Stream.__init__` for parameter
        details.

        :returns: A new :py:class:`PyAudio.Stream`
        """
        stream = PyAudio.Stream(self, *args, **kwargs)
        self._streams.add(stream)
        return stream

    def close(self, stream):
        """Closes a stream. Use :py:func:`PyAudio.Stream.close` instead.

        :param stream: An instance of the :py:class:`PyAudio.Stream` object.
        :raises ValueError: if stream does not exist.
        """
        if stream not in self._streams:
            raise ValueError(f"Stream {stream} not found")

        stream.close()

    def _remove_stream(self, stream):
        """Removes a stream. (Internal)

        :param stream: An instance of the :py:class:`PyAudio.Stream` object.
        """
        if stream in self._streams:
            self._streams.remove(stream)

    # Host API Inspection

    def get_host_api_count(self):
        """Returns the number of available PortAudio Host APIs.

        :rtype: integer
        """
        return pa.get_host_api_count()

    def get_default_host_api_info(self):
        """Returns a dictionary containing the default Host API parameters.

        The keys of the dictionary mirror the data fields of PortAudio's
        ``PaHostApiInfo`` structure.

        :raises IOError: if no default input device is available
        :rtype: dict
        """
        default_host_api_index = pa.get_default_host_api()
        return self.get_host_api_info_by_index(default_host_api_index)

    def get_host_api_info_by_type(self, host_api_type):
        """Returns a dictionary containing the Host API parameters for the
        host API specified by the `host_api_type`. The keys of the
        dictionary mirror the data fields of PortAudio's ``PaHostApiInfo``
        structure.

        :param host_api_type: The desired |PaHostAPI|
        :raises IOError: for invalid `host_api_type`
        :rtype: dict
        """
        index = pa.host_api_type_id_to_host_api_index(host_api_type)
        return self.get_host_api_info_by_index(index)

    def get_host_api_info_by_index(self, host_api_index):
        """Returns a dictionary containing the Host API parameters for the
        host API specified by the `host_api_index`. The keys of the
        dictionary mirror the data fields of PortAudio's ``PaHostApiInfo``
        structure.

        :param host_api_index: The host api index
        :raises IOError: for invalid `host_api_index`
        :rtype: dict
        """
        return self._make_host_api_dictionary(
            host_api_index,
            pa.get_host_api_info(host_api_index))

    def get_device_info_by_host_api_device_index(self,
                                                 host_api_index,
                                                 host_api_device_index):
        """Returns a dictionary containing the Device parameters for a
        given Host API's n'th device. The keys of the dictionary
        mirror the data fields of PortAudio's ``PaDeviceInfo`` structure.

        :param host_api_index: The Host API index number
        :param host_api_device_index: The n'th device of the host API
        :raises IOError: for invalid indices
        :rtype: dict
        """
        long_method_name = pa.host_api_device_index_to_device_index
        device_index = long_method_name(host_api_index, host_api_device_index)
        return self.get_device_info_by_index(device_index)

    def _make_host_api_dictionary(self, index, host_api_struct):
        """Creates dictionary like PortAudio's ``PaHostApiInfo`` structure.

        :rtype: dict
        """
        return {
            'index': index,
            'structVersion': host_api_struct.structVersion,
            'type': host_api_struct.type,
            'name': host_api_struct.name,
            'deviceCount': host_api_struct.deviceCount,
            'defaultInputDevice': host_api_struct.defaultInputDevice,
            'defaultOutputDevice': host_api_struct.defaultOutputDevice
        }

    # Device Inspection

    def get_device_count(self):
        """Returns the number of PortAudio Host APIs.

        :rtype: integer
        """
        return pa.get_device_count()

    def is_format_supported(self, rate,
                            input_device=None,
                            input_channels=None,
                            input_format=None,
                            output_device=None,
                            output_channels=None,
                            output_format=None):
        """Checks if specified device configuration is supported.

        Returns True if the configuration is supported; raises ValueError
        otherwise.

        :param rate:
           Specifies the desired rate (in Hz)
        :param input_device:
           The input device index. Specify ``None`` (default) for
           half-duplex output-only streams.
        :param input_channels:
           The desired number of input channels. Ignored if
           `input_device` is not specified (or ``None``).
        :param input_format:
           PortAudio sample format constant defined
           in this module
        :param output_device:
           The output device index. Specify ``None`` (default) for
           half-duplex input-only streams.
        :param output_channels:
           The desired number of output channels. Ignored if
           `input_device` is not specified (or ``None``).
        :param output_format:
           |PaSampleFormat| constant.

        :rtype: bool
        :raises ValueError: tuple containing (error string, |PaErrorCode|).
        """
        if input_device is None and output_device is None:
            raise ValueError(
                "Must specify stream format for input, output, or both",
                paInvalidDevice)

        kwargs = {}
        if input_device is not None:
            kwargs['input_device'] = input_device
            kwargs['input_channels'] = input_channels
            kwargs['input_format'] = input_format

        if output_device is not None:
            kwargs['output_device'] = output_device
            kwargs['output_channels'] = output_channels
            kwargs['output_format'] = output_format

        return pa.is_format_supported(rate, **kwargs)

    def get_default_input_device_info(self):
        """Returns the default input device parameters as a dictionary.

        The keys of the dictionary mirror the data fields of PortAudio's
        ``PaDeviceInfo`` structure.

        :raises IOError: No default input device available.
        :rtype: dict
        """
        device_index = pa.get_default_input_device()
        return self.get_device_info_by_index(device_index)

    def get_default_output_device_info(self):
        """Returns the default output device parameters as a dictionary.

        The keys of the dictionary mirror the data fields of PortAudio's
        ``PaDeviceInfo`` structure.

        :raises IOError: No default output device available.
        :rtype: dict
        """
        device_index = pa.get_default_output_device()
        return self.get_device_info_by_index(device_index)

    def get_device_info_by_index(self, device_index):
        """Returns the device parameters for device specified in `device_index`
        as a dictionary. The keys of the dictionary mirror the data fields of
        PortAudio's ``PaDeviceInfo`` structure.

        :param device_index: The device index
        :raises IOError: Invalid `device_index`.
        :rtype: dict
        """
        return self._make_device_info_dictionary(
            device_index,
            pa.get_device_info(device_index))

    def _make_device_info_dictionary(self, index, device_info):
        """Creates a dictionary like PortAudio's ``PaDeviceInfo`` structure.

        :rtype: dict
        """
        device_name = device_info.name

        # Attempt to decode device_name. If we fail to decode, return the raw
        # bytes and let the caller deal with the encoding.
        os_encoding = locale.getpreferredencoding(do_setlocale=False)
        for codec in [os_encoding, "utf-8"]:
            try:
                device_name = device_name.decode(codec)
                break
            except:
                pass

        return {'index': index,
                'structVersion': device_info.structVersion,
                'name': device_name,
                'hostApi': device_info.hostApi,
                'maxInputChannels': device_info.maxInputChannels,
                'maxOutputChannels': device_info.maxOutputChannels,
                'defaultLowInputLatency':
                device_info.defaultLowInputLatency,
                'defaultLowOutputLatency':
                device_info.defaultLowOutputLatency,
                'defaultHighInputLatency':
                device_info.defaultHighInputLatency,
                'defaultHighOutputLatency':
                device_info.defaultHighOutputLatency,
                'defaultSampleRate':
                device_info.defaultSampleRate}


# Host Specific Stream Info

if hasattr(pa, 'paMacCoreStreamInfo'):
    class PaMacCoreStreamInfo(pa.paMacCoreStreamInfo):
        """PortAudio Host API Specific Stream Info for macOS-specific settings.

        To configure macOS-specific settings, instantiate this class and pass
        it as the argument in :py:func:`PyAudio.open` to parameters
        ``input_host_api_specific_stream_info`` or
        ``output_host_api_specific_stream_info``.  (See
        :py:func:`PyAudio.Stream.__init__`.)

        :note: macOS-only.

        .. |PaMacCoreFlags| replace:: :ref:`PortAudio Mac Core Flags <PaMacCoreFlags>`
        .. _PaMacCoreFlags:

        **PortAudio Mac Core Flags**
          :py:data:`paMacCoreChangeDeviceParameters`,
          :py:data:`paMacCoreFailIfConversionRequired`,
          :py:data:`paMacCoreConversionQualityMin`,
          :py:data:`paMacCoreConversionQualityMedium`,
          :py:data:`paMacCoreConversionQualityLow`,
          :py:data:`paMacCoreConversionQualityHigh`,
          :py:data:`paMacCoreConversionQualityMax`,
          :py:data:`paMacCorePlayNice`,
          :py:data:`paMacCorePro`,
          :py:data:`paMacCoreMinimizeCPUButPlayNice`,
          :py:data:`paMacCoreMinimizeCPU`

        .. attribute:: flags

           The flags specified to the constructor.

           :type: |PaMacCoreFlags|

        .. attribute:: channel_map

           The channel_map specified to the constructor

           :type: tuple or None if unspecified
        """
        paMacCoreChangeDeviceParameters = pa.paMacCoreChangeDeviceParameters
        paMacCoreFailIfConversionRequired = pa.paMacCoreFailIfConversionRequired
        paMacCoreConversionQualityMin = pa.paMacCoreConversionQualityMin
        paMacCoreConversionQualityMedium = pa.paMacCoreConversionQualityMedium
        paMacCoreConversionQualityLow = pa.paMacCoreConversionQualityLow
        paMacCoreConversionQualityHigh = pa.paMacCoreConversionQualityHigh
        paMacCoreConversionQualityMax = pa.paMacCoreConversionQualityMax
        paMacCorePlayNice = pa.paMacCorePlayNice
        paMacCorePro = pa.paMacCorePro
        paMacCoreMinimizeCPUButPlayNice = pa.paMacCoreMinimizeCPUButPlayNice
        paMacCoreMinimizeCPU = pa.paMacCoreMinimizeCPU

        def __init__(self, flags=None, channel_map=None):
            """Initialize with macOS setting flags and channel_map.

            See PortAudio documentation for more details on these parameters.

            :param flags: |PaMacCoreFlags| OR'ed together.
            :param channel_map: An array describing the channel mapping.
                See PortAudio documentation for usage.
            """
            kwargs = {}
            if flags is not None:
                kwargs["flags"] = flags
            if channel_map is not None:
                kwargs["channel_map"] = channel_map
            super().__init__(**kwargs)

        # Deprecated:

        def get_flags(self):
            """Returns the flags set at instantiation. Deprecated.

            :rtype: integer

            .. deprecated:: 0.2.13
               Use :py:attr:`flags` property.
            """
            warnings.warn(
                "PaMacCoreStreamInfo.get_flags is deprecated. Use the flags "
                "property instead.",
                DeprecationWarning,
                stacklevel=2)
            return self.flags

        def get_channel_map(self):
            """Returns the channel map set at instantiation. Deprecated.

            :rtype: tuple or None

            .. deprecated:: 0.2.13
               Use :py:attr:`channel_map` property.
            """
            warnings.warn(
                "PaMacCoreStreamInfo.get_channel_map is deprecated. Use the "
                "channel_map property instead.",
                DeprecationWarning,
                stacklevel=2)
            return self.channel_map

        def _get_host_api_stream_object(self):
            """Returns the underyling stream info.

            .. :deprecated:: 0.2.13
               Use stream_info property.
            """
            warnings.warn(
                "PaMacCoreStreamInfo._get_host_api_stream_object is "
                "deprecated. Use this object instance instead.",
                DeprecationWarning,
                stacklevel=2)
            return self


# The top-level Stream class is reserved for future API changes. Users should
# never instantiate Stream directly. Instead, users must use PyAudio.open()
# instead, as documented.
#
# But for existing code that happens to instantiate Stream directly, this class
# issues a warning and maintains backwards-compatibility, for now. In the
# future, Stream may be repurposed.
class Stream(PyAudio.Stream):
    """Reserved. Do not instantiate."""

    def __init__(self, *args, **kwargs):
        # Users should never instantiate this class.
        warnings.warn(
            "Do not instantiate pyaudio.Stream directly. Use "
            "pyaudio.PyAudio.open() instead. pyaudio.Stream may change or be "
            "removed in the future.",
            DeprecationWarning,
            stacklevel=2)
        super().__init__(*args, **kwargs)
