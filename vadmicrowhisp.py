#!/usr/bin/env python3

import sys
import os
from pathlib import Path
import argparse
import csv
import asyncio
import logging
import yaml
import wave
import time
import torch
import resampy
import numpy as np
import queue
import traceback
from threading import Thread
import requests

from dataclasses import is_dataclass, asdict
from faster_whisper import WhisperModel

import json
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion

import gstmicpipeline as gm

from vad_iterator import VADIterator

# configure logger
logging.basicConfig(
    format="%(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO)
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

scrroot = Path('.')

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


def current_milli_time(audio_time):
    """Return current (unix) time in milliseconds.

    The audio_time argument is deliberately ignored.
    """
    return round(time.time() * 1000)


def audio_milli_time(audio_time):
    """ audio time is processed audio in seconds """
    return round(audio_time * 1000)


def init_jit_model(model_path: str, device=torch.device('cpu')):
    logger.info("Init VAD model")
    torch.set_grad_enabled(False)
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model


def named_tupel_to_dictionary(tupel):
    """
    Convert a named tuple into a dictionary. Use nested dictionaries if tuple value is another tupel.
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
    """ Monitor input to .wav file, Takes path, sample rate, an no. channels
    """
    wf = wave.open(path, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    return wf


class WhisperMicroServer():
    MAX_BUF_RETENTION = 40
    MIN_SPEECH_DETECTS = 3
    MIN_SILENCE_DETECTS = 30
    BUFFER_SIZE = 512

    def __init__(self, config, micro=True, transcription_file=None):
        self.pid = "whisperasr"
        self.topics = {}  # string to fn or (fn, qos)

        self.audio_dir = "audio/"
        self.language = "de"

        self.channels = 1
        self.usedchannel = 0
        self.sample_rate = 16000
        self.asr_sample_rate = 16000
        self.buffers_queued = 6

        self.loop = None
        self.is_running = True
        self.from_micro = micro

        if self.from_micro:
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
        self.transcription_queue = queue.Queue(maxsize=1000)
        self.initial_prompt = ''
        self.client = None
        self.transcription_file = transcription_file
        self.__init_mqtt_client()
        # create 100 ms buffer with silence (2 bytes per sample): / 1000 * 100
        self.silence_buffer = bytearray(WhisperMicroServer.BUFFER_SIZE *
                                        (self.buffers_queued + 1))
        # load silero VAD model
        model = init_jit_model(model_path=scrroot/'models'/'silero_vad.jit')

        vad_config = config['vad'] if 'vad' in config else dict()
        #print(type(vad_config['threshold']))
        self.vad_iterator = VADIterator(model, **vad_config)
        #self.threshold = 0.5
        #print(f'{self.asr_sample_rate} {self.sample_rate} {self.channels}')

        self.__init_whisper()

        # for monitoring (eventually)
        self.am = None
        self.wf = None

    def __init_whisper(self):
        self.whisper_model = None
        if 'whisper' not in self.config:
            logger.error('no whisper config section: minimally specify model size or URL')
            sys.exit(1)
        whisper_config = self.config['whisper']
        self.whisper_url = whisper_config.get('url', None)
        if self.whisper_url:
            return   # using remote ASR
        if 'device' not in whisper_config or whisper_config['device'] == 'cpu':
            whisper_config['device'] = 'cpu'
            if not 'compute_type' in whisper_config:
                whisper_config['compute_type'] = 'int16'
        else:
            whisper_config['device'] = 'cuda'
            if not 'compute_type' in whisper_config:
                whisper_config['compute_type'] = 'float32'
        if 'whisper_transcription' not in self.config:
            self.config['whisper_transcription'] = dict()
        if self.language and not 'language' in self.config['whisper_transcription']:
            self.config['whisper_transcription']['language'] = self.language
        logger.info(
            f"initializing {whisper_config['model_size']} model "
            f"for {whisper_config['device']} {whisper_config['compute_type']} ...")
        model_path = scrroot / 'models' / 'whisper' / whisper_config['model_size']
        self.whisper_model = WhisperModel(str(model_path) + '/',
                                          device=whisper_config['device'],
                                          compute_type=whisper_config['compute_type'])
        logger.info("Whisper model initialized")

    def __init_transcription_thread(self):
        logger.info("start transcription thread...")
        self.transcribe_thread = Thread(target=self.transcribe,
                                        daemon=self.from_micro)
        self.transcribe_thread.start()
        logger.info("transcription thread running")

    def __init_mqtt_client(self):
        self.client = mqtt.Client(CallbackAPIVersion.VERSION2)
        # self.client.username_pw_set(self.mqtt_username, self.mqtt_password)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.prompt_topic = self.pid + '/set_prompt'
        self.topics[self.prompt_topic] = self._on_prompt_msg

    def _on_prompt_msg(self, client, userdata, message):
        self.initial_prompt = message.payload
        logger.info(f'new prompt: {self.initial_prompt}')

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

    def wav_filename(self):
        return self.audio_dir + f'source-{current_milli_time(0):014d}.wav'

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
        self.client.connect(self.config['mqtt_address'])
        self.client.loop_start()

    def mqtt_disconnect(self):
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()

    def send_transcription(self, trans: dict):
        if self.client:
            self.client.publish(self.topic, json.dumps(trans, indent=None))
        if self.transcription_file:
            # massage the information in trans into the right format
            text = ""
            for segment in trans['segments']:
                text += segment['text'] + ' '
            trans.pop('info')
            trans.pop('segments')
            trans['text'] = text.strip()
            if self.inputfile:
                trans['filename'] = self.inputfile
            self.transcription_file.writerow(trans)

    def transcribe(self):
        """
        Monitor transcription queue for incoming audio to be transcribed with
        local Whisper and add resulting transcriptions to result list
        :return:
        """

        while self.is_running or not self.transcription_queue.empty():
            # this call blocks until an element can be retrieved from queue
            audio_segment, start, end = self.transcription_queue.get()
            conv_params = self.config['whisper_transcription']
            #if 'initial_prompt' not in conv_params and self.prompt:
            if self.initial_prompt:
                conv_params['initial_prompt'] = self.initial_prompt
            if not self.whisper_url:
                try:
                    logger.info("now transcribing")
                    segments, info = self.whisper_model.transcribe(np.array(audio_segment), **conv_params)
                    logger.info("transcribing...")
                    transcripts = []
                    for segment in segments:
                        logger.info("[%.2fs -> %.2fs] %s"
                                    % (segment.start, segment.end, segment.text))
                        conv_dict = named_tupel_to_dictionary(segment)
                        transcripts.append(conv_dict)
                    res = {'info': named_tupel_to_dictionary(info),
                           'segments': transcripts,
                           'start': start, 'end': end}
                    #res.update(self.speaker_identification(audio_segment))
                    self.send_transcription(res)

                except Exception as ex:
                    logger.error('whisper exception: {}'.format(ex))
                    traceback.print_exc()
                    del self.whisper_model.model
                    del self.whisper_model
                    self.__init_whisper()
            else:
                segment = np.array(audio_segment, dtype=np.float32)
                segment /= 32768
                response = requests.post(url=self.whisper_url,
                                         data=segment.tobytes(),
                                         headers={'Content-Type': 'application/octet-stream'},
                                         params=conv_params)
                remote_result = response.json()
                remote_result['start'] = start
                remote_result['end'] = end
                #remote_result.update(self.speaker_identification(audio_segment))
                self.send_transcription(remote_result)
                if 'segments' in remote_result:
                    for segment in remote_result['segments']:
                        logger.info("[%.2fs -> %.2fs] %s"
                                    % (segment['start'], segment['end'], segment['text']))

        logger.info("Leaving transcribe")

    def bytes2intlist(self, audio):
        frame = np.frombuffer(audio, dtype=np.int16)
        # monitor what comes in
        # data will be self.sample_rate, mono, np.int16 ndarray
        frame = self.resample(frame, self.channels, self.sample_rate)
        #print('d', np.shape(data))
        #print('vb1', len(voice_buffers))
        return frame.tolist()

    async def audio_loop(self):
        logger.info(f'sample_rate: {self.asr_sample_rate}')
        is_voice = -1
        window_size_samples = WhisperMicroServer.BUFFER_SIZE
        framesqueued = window_size_samples * self.buffers_queued
        voice_buffers = [0] * framesqueued
        out_buffer = []
        audio_time = 0.0
        while self.is_running or not self.audio_queue.empty():
            if len(voice_buffers) < window_size_samples + framesqueued:
                # this is a byte buffer
                audio = await self.audio_queue.get()
                # monitor what comes in
                if self.am:
                    self.am.writeframes(audio)
                voice_buffers.extend(self.bytes2intlist(audio))
                if len(voice_buffers) < window_size_samples + framesqueued:
                    continue

            ichunk = voice_buffers[framesqueued:framesqueued + window_size_samples]
            chunk = np.array(ichunk, dtype=np.int16)
            vadbuf = chunk / 32768
            speech_dict = self.vad_iterator(vadbuf, return_seconds=True)
            audio_time += len(ichunk) / self.asr_sample_rate
            if speech_dict:
                #print(f'{speech_dict}')
                if "start" in speech_dict:
                    # arg ist processed time in milliseconds
                    is_voice = self.timestamp_fn(audio_time)
                    print('<', end='', flush=True)
                    # monitor what is sent to the ASR
                    if self.config.get('monitor_asr', False):
                        # Always mono
                        self.wf = open_wave_file(
                            self.asrmon_filename(is_voice),
                            self.asr_sample_rate, 1)
                    # add queued buffers to the outbuffer
                    out_buffer = voice_buffers[:framesqueued]
                    audiodata = np.array(out_buffer, dtype=np.int16).tobytes()
                    self.writeframes(audiodata)
                elif "end" in speech_dict:
                    if is_voice < 0:
                        print('VAD end ignored')
                        break
                    print('>', end='', flush=True)
                    self.writeframes(chunk.tobytes())
                    self.writeframes(self.silence_buffer)
                    out_buffer.extend(ichunk)
                    self.transcription_queue.put(
                        (out_buffer, is_voice, self.timestamp_fn(audio_time)))
                    is_voice = -1
                    out_buffer = []
                    if self.wf:
                        self.wf.close()
                        self.wf = None
            voice_buffers = voice_buffers[window_size_samples:]
            if is_voice >= 0:
                self.writeframes(chunk.tobytes())
                out_buffer.extend(ichunk)
        logger.info("Leaving audio_loop")

    def cb(self, inp, frames):
        self.callback(inp, frames, None, None)

    async def run_micro(self):
        self.inputfile = None
        self.__init_transcription_thread()
        self.loop = asyncio.get_running_loop()
        pipeline = self.config["pipeline"] if "pipeline" in self.config \
            else gm.PIPELINE
        try:
            self.device = gm.GstreamerMicroSink(callback=self.cb,
                                                pipeline_spec=pipeline)
            self.device.start()
            if self.config.get('monitor_mic', False):
                self.am = open_wave_file(self.wav_filename(),
                                         self.sample_rate, self.channels)

            print("Connecting to MQTT broker")
            self.mqtt_connect()
            await self.audio_loop()
        finally:
            print('Disconnecting...')
            if self.am:
                self.am.close()
            self.mqtt_disconnect()

    def stop(self):
        self.is_running = False
        self.audio_queue.put_nowait(self.silence_buffer)
        self.device.stop()

    def read_file(self, file):
        self.is_running = False   # leave transcribe when queue empty
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
                    self.transcription_queue.put((buf, now, now + duration))
                    self.transcribe()
                    break

    def run(self, config, files, mqtt):
        if mqtt:
            print("Connecting to MQTT broker")
            self.mqtt_connect()
        #self.loop = asyncio.get_running_loop()
        #self.processing = asyncio.create_task(self.audio_loop())
        #self.processing.add_done_callback(
        #   lambda t: logger.info("processing loop finished"))
        for file in files:
            self.inputfile = os.path.splitext(os.path.basename(file))[0]
            logger.info("Processing {}".format(self.inputfile))
            self.read_file(file)

    def stop_batch(self):
        self.is_running = False
        print('Disconnecting...')
        if self.am:
            self.am.close()
        self.mqtt_disconnect()



def process_files(config_file, files, output_dir, mqtt):
    config = load_config(config_file)
    root = output_dir if output_dir else "outputs"
    if root[:-1] != os.sep:
        root += os.sep
    outdir = root
    if not os.path.exists(outdir):
        logger.info("Creating {}".format(outdir))
        os.makedirs(outdir)

    with open(outdir + 'batch.csv', 'w') as csvfile:
        fieldnames = ['filename', 'start', 'end', 'text',
                      'embedid', 'speaker', 'confidence']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        config['audio_dir'] = outdir + os.sep
        ms = WhisperMicroServer(config, micro=False, transcription_file=writer)
        ms.run(config, files, mqtt)


def load_config(file):
    with open(file, 'r') as f:
        return yaml.safe_load(f)


def main(config_file):
    #if len(args) < 1:
    #    sys.stderr.write('Usage: %s <config.yaml> [audio_file(s)]\n' % args[0])
    #    sys.exit(1)

    config = load_config(config_file)
    ms = WhisperMicroServer(config)

    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(ms.run_micro())
    except Exception as ex:
        logger.error("Exception: " + str(ex))
        traceback.print_exc()
        ms.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Whisper Server',
        description='Listen to microphone and transcribe input or analyse a set of files',
        epilog='')
    parser.add_argument("-c", "--config", metavar='config', type=str,
                        required=True, help='config file')
    parser.add_argument("-o", "--output-dir", metavar='output_dir', type=str,
                        required=False, help='output directory for chunks')
    parser.add_argument("-m", "--mqtt", action='store_true',
                        required=False, help='send mqtt messages in batch processing')
    parser.add_argument('files', metavar='files', type=str, nargs='*')
    args = parser.parse_args()
    scrroot = Path(sys.argv[0]).parent
    if (args.files):
        process_files(args.config, args.files, args.output_dir, args.mqtt)
    else:
        main(args.config)
