#!/usr/bin/env python3

import os
from pathlib import Path
import argparse
import csv
import asyncio
import logging
import yaml
import wave
import time
import resampy
import numpy as np
import queue
import traceback
from threading import Thread
import json

from dataclasses import is_dataclass, asdict

import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion

import gstmicpipeline as gm

from vad_iterator import VadState

MICRO="microphone"

# configure logger
logging.basicConfig(
    format="%(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO)
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

modroot = Path('.') / 'models'

# enable debugging at httplib level (requests->urllib3->http.client)
# You will see the REQUEST, including HEADERS and DATA, and RESPONSE
# with HEADERS but without DATA.
# The only thing missing will be the response.body which is not logged.

# import http.client as http_client
# http_client.HTTPConnection.debuglevel = 1
#
# requests_log = logging.getLogger("requests.packages.urllib3")
# requests_log.setLevel(logging.DEBUG)
# requests_log.propagate = True


def int_or_str(text):
    """Try to convert to int, return original object if not possible."""
    try:
        return int(text)
    except ValueError:
        return text


def current_milli_time():
    """
    Return the unix time in milliseconds.

    The audio_time argument is deliberately ignored!
    """
    return round(time.time() * 1000)


def audio_milli_time(audio_time):
    """Audio time is processed audio in seconds."""
    return round(audio_time * 1000)


def named_tupel_to_dictionary(tupel):
    """
    Convert a named tuple into a dictionary. Use nested dictionaries if tuple
    value is another tupel.
    :param tupel: named tupel
    :return: named tupel converted into dictionary
    """

    result_dict = {}
    for key, value in asdict(tupel).items():
        if is_dataclass(value):
            result_dict[key] = named_tupel_to_dictionary(value)
        elif isinstance(value, list):
            conv_list = []
            for item in value:
                if is_dataclass(item):
                    conv_list.append(named_tupel_to_dictionary(item))
                elif isinstance(item, tuple):
                    # special handling of language prob lists
                    conv_list.append({'lang': item[0], 'prob': item[1]})
                else:
                    conv_list.append(item)
            if conv_list:
                result_dict[key] = conv_list
        elif value is None:
            pass
        else:
            result_dict[key] = value
    return result_dict


def open_wave_file(path, sample_rate, channels):
    """ Monitor input to .wav file, Takes path, sample rate, an no. channels.
    """
    wf = wave.open(path, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    return wf


class Transkriptor():
    MAX_BUF_RETENTION = 40
    MIN_SPEECH_DETECTS = 3
    MIN_SILENCE_DETECTS = 30
    BUFFER_SIZE = 512

    def __init__(self, config, pid, transcription_file=None):
        self.pid = pid
        self.topics = {
            self.pid + '/control': self._on_control_msg,
        }

        self.audio_dir = "audio/"
        self.language = ""
        self.encoding = "utf-8"

        self.channels = 1
        self.usedchannel = 0
        self.sample_rate = 16000
        self.asr_sample_rate = 16000
        self.buffers_queued = 6

        self.loop: asyncio.AbstractEventLoop
        self.is_running = True
        self.audio_source = MICRO
        self.device = None
        self.always_use_vad = False

        if self.from_micro():
            self.timestamp_fn = current_milli_time
            self.audio_queue = asyncio.Queue()
        else:
            self.timestamp_fn = audio_milli_time
            self.audio_queue = asyncio.Queue(maxsize=1)

        self.config = config
        if 'asr_sample_rate' in config:
            self.asr_sample_rate = config['asr_sample_rate']
        if 'sample_rate' in config:
            self.sample_rate = config['sample_rate']
        if 'channels' in config:
            self.channels = config['channels']
        if 'use_channel' in config:
            self.usedchannel = config['use_channel']
        if 'audio_dir' in config:
            self.audio_dir = config['audio_dir']
        if 'language' in config:
            self.language = config['language']
        if 'buffers_queued' in config:
            self.buffers_queued = config['buffers_queued']
        self.topic = self.pid + '/asrresult'
        if self.language:
            self.topic += '/' + self.language
        self.initial_prompt = ''
        self.transcription_file = transcription_file
        self.__init_mqtt_client()
        # create 2s buffer with silence (2 bytes per sample)
        self.silence_buffer = bytearray(self.asr_sample_rate * 2)
        # initialize silero VAD model
        vad_config = config.get('vad', dict())
        self.vad_state = VadState(modroot / 'silero_vad.jit', **vad_config)
        self.start_time = None

        self.transcription_queue = queue.Queue(maxsize=1000)
        self.__init_transcription_thread()

        # for monitoring (eventually)
        self.am = None
        self.wf = None


    valid_mqtt_keys = {'host', 'port', 'keepalive', 'bind_address', 'bind_port'
                        'clean_start'}

    def __init_mqtt_client(self):
        if 'mqtt' not in self.config:
            self.config['mqtt'] = { 'host':'localhost' }
        mqtt_config = self.config['mqtt']
        for key in mqtt_config:
            if key not in self.__class__.valid_mqtt_keys:
                del(key, mqtt_config)
        self.client: mqtt.Client
        self.client = mqtt.Client(CallbackAPIVersion.VERSION2)
        if 'username' in mqtt_config and 'password' in mqtt_config:
            self.client.username_pw_set(mqtt_config['username'],
                                        mqtt_config['password'])
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    def __init_transcription_thread(self):
        logger.info("start transcription thread...")
        self.transcribe_thread = Thread(target=self.transcribe_async,
                                        daemon=self.from_micro())
        self.transcribe_thread.start()
        logger.info("transcription thread running")

    def _pause(self):
        if self.from_micro() and self.device:
            self.device.pause()

    def _unpause(self):
        if self.from_micro() and self.device:
            self.device.start()

    def _on_prompt_msg(self, client, userdata, message):
        self.initial_prompt = message.payload.decode(self.encoding)
        logger.info(f'new prompt: {self.initial_prompt}')

    def _on_control_msg(self, client, userdata, message):
        message = message.payload.decode(self.encoding)
        logger.info(f'control message: {message}')
        match message:
            case 'exit':
                self.is_running = False
            case 'pause_mic':
                self._pause()
            case 'unpause_mic':
                self._unpause()
            case _:
                if message.startswith('process_file'):
                    filename = message.split(':')[1]
                    self.transcription_queue.put(filename)

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        logger.debug(f'CONNACK received with code {reason_code}')
        # subscribe to all registered topics/callbacks
        for topic in self.topics:
            qos = 0
            if topic is tuple:
                qos = topic[1]
                topic = topic[0]
            self.client.subscribe(topic, qos)

    def _on_message(self, client, userdata, message):
        logger.debug(f"Received message {str(message.payload)} on topic {message.topic} with QoS {str(message.qos)}")
        if message.topic not in self.topics:
            self.topics[message.topic] = None
            for topic in self.topics:
                if mqtt.topic_matches_sub(topic, message.topic):
                    self.topics[message.topic] = self.topics[topic]
        cb = self.topics[message.topic]
        if cb is not None:
            if cb is tuple:
                cb = cb[0]  # second is qos
            cb(client, userdata, message)
        return

    def from_micro(self):
        return self.audio_source == MICRO

    def wav_filename(self):
        return self.audio_dir + f'source-{current_milli_time():014d}.wav'

    def asrmon_filename(self, suffix):
        return self.audio_dir + f'chunk-{suffix:014}.wav'

    def writeframes(self, audio):
        if self.wf:
            self.wf.writeframes(audio)

    def resample(self, frame, channels, sample_rate):
        if channels > 1:
            # numpy slicing:
            # take every i'th value: frame[start:stop:step]
            frame = frame[self.usedchannel::channels]
        if sample_rate != self.asr_sample_rate:
            frame = resampy.resample(frame, sample_rate, self.asr_sample_rate)
            frame = frame.astype(np.int16)
        return frame

    def callback(self, indata, frames, time_block, status):
        """This is called (from a separate thread) for each audio block."""
        self.loop.call_soon_threadsafe(self.audio_queue.put_nowait,
                                       bytes(indata))

    def mqtt_connect(self):
        self.client.connect(**self.config['mqtt'])
        self.client.loop_start()

    def mqtt_disconnect(self):
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()

    def bytes2intlist(self, audio):
        frame = np.frombuffer(audio, dtype=np.int16)
        # monitor what comes in
        # data will be self.sample_rate, mono, np.int16 ndarray
        frame = self.resample(frame, self.channels, self.sample_rate)
        # print('d', np.shape(data))
        # print('vb1', len(voice_buffers))
        return frame.tolist()

    def intlist2bytes(self, buffer):
        return np.array(buffer, dtype=np.int16).tobytes()

    def add_audio(self, audio):
        return self.vad_state.add_audio(self.bytes2intlist(audio))

    def send_transcription(self, trans: dict):
        if self.client:
            self.client.publish(self.topic, json.dumps(trans, indent=None))
        if self.transcription_file:
            # massage the information in trans into the right format
            trans.pop('info')
            trans.pop('segments')
            trans['source'] = self.audio_source
            self.transcription_file.writerow(trans)

    def transcribe_success(self, result, audio_segment):
        if 'text' in result:
            logger.info(f"text: {result['text']}")
            for segment in (result['segments'] if 'segments' in result else []):
                logger.info("[%.2fs -> %.2fs] %s"
                            % (segment['start'], segment['end'],
                               segment['text']))
        self.send_transcription(result)

    def transcribe_async(self):
        """
        Monitor transcription queue for incoming audio to be transcribed with
        local Whisper and add resulting transcriptions to result list.
        """
        while self.is_running:
            # this call blocks until an element can be retrieved from queue
            to_transcribe = self.transcription_queue.get()
            if isinstance(to_transcribe, tuple):
                audio_segment, start, end = to_transcribe
                self.transcribe(audio_segment, start, end)
            else: # a string representing a file to process
                if to_transcribe.endswith('.csv'):
                    self.process_dataset(Path(to_transcribe))
                else:
                    # must be a readable audio file!
                    self.process_file(to_transcribe)

        logger.info("Leaving async transcribe")

    def transcribe(self, audio_segment, start, end):
        """To be implemented by subclasses."""
        pass

    def write_buffer(self, buffer):
        buf = self.intlist2bytes(buffer)
        if self.wf:
            self.wf.writeframes(buf)
        return buf

    def start_buffer(self, buffer):
        """To be extended by subclasses."""
        self.start_time = current_milli_time()
        if self.config.get('monitor_asr', False):
            self.wf = open_wave_file(self.asrmon_filename(self.start_time),
                                     self.asr_sample_rate, 1)
        return self.write_buffer(buffer)

    def continue_buffer(self, buffer):
        """To be extended by subclass."""
        return self.write_buffer(buffer)

    def end_buffer(self, buffer, end_time):
        """To be extended by subclass."""
        buf = self.write_buffer(buffer)
        if self.wf:
            self.wf.close()
            self.wf = None
        self.start_time = None
        return buf

    async def microphone_loop(self):
        """Wait for audio segments to come in and process them"""
        logger.info(f'sample_rate: {self.asr_sample_rate}')
        while self.is_running or not self.audio_queue.empty():
            audio = await self.audio_queue.get()
            if self.am:
                self.am.writeframes(audio)
            state, buffer = self.add_audio(audio)
            if state == "start":
                self.start_buffer(buffer)
            elif state == "continue":
                self.continue_buffer(buffer)
            elif state == "end":
                self.end_buffer(buffer, current_milli_time())
        logger.info("Leaving audio_loop")

    def cb(self, inp, frames):
        self.callback(inp, frames, None, None)

    async def run_micro(self):
        """ TODO: REFACTOR"""
        self.inputfile = None
        self.loop = asyncio.get_running_loop()
        pipeline = self.config["pipeline"] if "pipeline" in self.config \
            else gm.PIPELINE
        try:
            self.device = gm.GstreamerMicroSink(callback=self.cb,
                                                pipeline_spec=pipeline)
            self.device.start()
            logger.info("Connecting to MQTT broker")
            self.mqtt_connect()
            if self.config.get('monitor_mic', False):
                with open_wave_file(self.wav_filename(),
                                    self.sample_rate, self.channels) as self.am:
                    await self.microphone_loop()
            else:
                self.am = None
                await self.microphone_loop()
        finally:
            self.device.stop()
            logger.info('Disconnecting...')
            self.mqtt_disconnect()

    def stop(self):
        self.is_running = False
        self.audio_queue.put_nowait(self.silence_buffer)

    def read_file(self, file):
        """Synchronously read the file and transcribe the audio."""
        with wave.open(file, "rb") as wf:
            self.channels = wf.getnchannels()
            self.sample_rate = wf.getframerate()
            buffer_size = int(self.sample_rate * 0.2)  # 0.2 seconds of audio
            filebuf = bytearray()
            while True:
                data = wf.readframes(buffer_size)
                if len(data) > 0:
                    filebuf.extend(data)
                else:
                    # all timestamps in milliseconds
                    now = int(time.time() * 1000)
                    buf = self.bytes2intlist(filebuf)
                    # milliseconds since sample_rate is Hz
                    duration = int(float(len(buf) * 1000) /
                                   (self.sample_rate * self.channels))
                    self.transcribe(buf, now, now + duration)
                    break

    def read_file_vad(self, file):
        with wave.open(file, "rb") as wf:
            self.channels = wf.getnchannels()
            self.sample_rate = wf.getframerate()
            buffer_size = 512

            # samples --> milliseconds since sample_rate is Hz
            sample2time = 1000.0 / (self.sample_rate * self.channels)

            processed_samples = 0
            active = True
            start_time = 0
            while active:
                data = wf.readframes(buffer_size)
                if len(data) == 0:
                    data = self.silence_buffer
                    active = False
                state, buffer = self.add_audio(data)
                if state == "start":
                    start_time = int(processed_samples * sample2time)
                elif state == "continue":
                    pass
                elif state == "end":
                    buffer = self.vad_state.last_segment()
                    duration = int(len(buffer) * sample2time)
                    self.transcribe(buffer, start_time, start_time + duration)
                processed_samples += len(data)

    def process_file(self, file, do_pausing=True):
        # self.loop = asyncio.get_running_loop()
        # self.processing = asyncio.create_task(self.audio_loop())
        # self.processing.add_done_callback(
        #    lambda t: logger.info("processing loop finished"))
        curr_audio = (self.audio_source, self.channels, self.sample_rate)
        try:
            if do_pausing:
                self._pause()
            p = Path(file)
            self.audio_source = p.name
            logger.info("Processing {}".format(self.audio_source))
            p = str(p) # currently wave.open does not accept Path objects
            if self.always_use_vad:
                self.read_file_vad(p)
            else:
                self.read_file(p)
        except Exception as ex:
            logger.error(f"Error reading wav file: {ex}")
        finally:
            self.audio_source, self.channels, self.sample_rate = curr_audio
            if do_pausing:
                self._unpause()

    def process_dataset(self, csvfilepath: Path):
        try:
            filedir = csvfilepath.parent
            self._pause()
            with open(csvfilepath) as csvfile:
                reader = csv.DictReader(csvfile, delimiter=',')
                for row in reader:
                    self.process_file(filedir / row['file_name'], False)
        except Exception as ex:
            logger.error(f"Error reading csv file: {ex}")
        finally:
            self._unpause()

    def stop_batch(self):
        self.is_running = False
        print('Disconnecting...')
        if self.am:
            self.am.close()
        self.mqtt_disconnect()


def process_files(server_class, config_file, files, output_dir, mqtt, vad):
    config = load_config(config_file)
    root = output_dir if output_dir else "outputs"
    if root[:-1] != os.sep:
        root += os.sep
    outdir = root
    if not os.path.exists(outdir):
        logger.info("Creating {}".format(outdir))
        os.makedirs(outdir)

    with open(outdir + 'batch.csv', 'w') as csvfile:
        fieldnames = ['source', 'start', 'end', 'text',
                      'embedid', 'speaker', 'confidence']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        config['audio_dir'] = outdir + os.sep
        ms = server_class(config, transcription_file=writer)
        ms.always_use_vad = vad
        if mqtt:
            print("Connecting to MQTT broker")
            ms.mqtt_connect()
        for file in files:
            ms.process_file(file)


def load_config(file):
    with open(file, 'r') as f:
        return yaml.safe_load(f)


def main(server_class, prog_name):
    parser = argparse.ArgumentParser(
        prog=prog_name,
        description='Listen to microphone and transcribe input ' +
        'or analyse a set of files',
        epilog='')
    parser.add_argument("-c", "--config", metavar='config', type=str,
                        required=True, help='config file')
    parser.add_argument("-o", "--output-dir", metavar='output_dir', type=str,
                        required=False, help='output directory for chunks')
    parser.add_argument("-m", "--mqtt", action='store_true',
                        required=False,
                        help='send mqtt messages in batch processing')
    parser.add_argument("-v", "--vad", action='store_true',
                        required=False,
                        help='use VAD for file processing')
    parser.add_argument('files', metavar='files', type=str, nargs='*')
    args = parser.parse_args()
    if (args.files):
        process_files(server_class,
                      args.config, args.files, args.output_dir,
                      args.mqtt, args.vad)
    else:
        config = load_config(args.config)
        ms = server_class(config)
        ms.always_use_vad = args.vad

        logging.basicConfig(level=logging.INFO)
        try:
            asyncio.run(ms.run_micro())
        except Exception as ex:
            logger.error("Exception: " + str(ex))
            traceback.print_exc()
            ms.stop()
