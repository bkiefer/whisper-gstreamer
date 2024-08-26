#!/usr/bin/env python3

import sys
import os
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
from threading import Thread

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



def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

def current_milli_time(audio_time):
    return round(time.time() * 1000)

def audio_milli_time(audio_time):
    """ audio time is processed audio in seconds """
    return round(audio_time * 1000)

def init_jit_model(model_path: str, device=torch.device('cpu')):
    torch.set_grad_enabled(False)
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model

def isinstance_namedtuple(obj):
    return (
            isinstance(obj, tuple) and
            hasattr(obj, '_asdict') and
            hasattr(obj, '_fields')
    )

def named_tupel_to_dictionary(tupel):
    """
    Convert a named tuple into a dictionary. Use nested dictionaries if tuple value is another tupel.
    :param tupel: named tupel
    :return: named tupel converted into dictionary
    """
    result_dict = {}
    for key, value in tupel._asdict().items():
        if isinstance_namedtuple(value):
            result_dict[key] = named_tupel_to_dictionary(value)
        elif isinstance(value, list):
            conv_list = []
            for item in value:
                if isinstance_namedtuple(item):
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

class WhisperMicroServer():

    MAX_BUF_RETENTION = 40
    MIN_SPEECH_DETECTS = 3
    MIN_SILENCE_DETECTS = 30
    BUFFER_SIZE = 512

    def __init__(self, config, micro=True, transcription_file=None):
        self.pid = "whisperasr"
        self.audio_dir = "audio/"
        self.language = "de"

        self.channels = 1
        self.usedchannel = 0
        self.sample_rate = 16000
        self.asr_sample_rate = 16000
        self.buffers_queued = 6

        if micro:
            self.timestamp_fn = current_milli_time
            self.audio_queue = asyncio.Queue()
        else:
            self.timestamp_fn = audio_milli_time
            self.audio_queue = asyncio.Queue(maxsize=1)

        self.loop = None
        self.is_running = True

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
        self.client = None
        self.transcription_file = transcription_file
        self.__init_mqtt_client()
        # create 100 ms buffer with silence (2 bytes per sample): / 1000 * 100
        self.silence_buffer = bytearray(WhisperMicroServer.BUFFER_SIZE * (self.buffers_queued + 1))
        # load silero VAD model
        model = init_jit_model(model_path='silero_vad.jit')

        vad_config = config['vad'] if 'vad' in config else dict()
        #print(type(vad_config['threshold']))
        self.vad_iterator = VADIterator(model, **vad_config)
        #self.threshold = 0.5
        #print(f'{self.asr_sample_rate} {self.sample_rate} {self.channels}')

        self.__init_whisper()
        self.__init_transcription_thread(micro)

        # for monitoring (eventually)
        self.am = None
        self.wf = None

    def __init_whisper(self):
        self.whisper_model = None
        if 'whisper' not in self.config:
            logger.error('no whisper config section: minimally specify model size')
            sys.exit(1)
        whisper_config = self.config['whisper']
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
        model_path = "./whisper-models/" + whisper_config['model_size'] + '/'
        self.whisper_model = WhisperModel(model_path,
                                          device=whisper_config['device'],
                                          compute_type=whisper_config['compute_type'])
        logger.info("Whisper model initialized")

    def __init_transcription_thread(self, daemon):
        logger.info("start transcription thread...")
        self.transcribe_thread = Thread(target=self.transcribe, daemon=daemon)
        self.transcribe_thread.start()
        logger.info("transcription thread running")

    def __init_mqtt_client(self):
        self.client = mqtt.Client(CallbackAPIVersion.VERSION2)
        # self.client.username_pw_set(self.mqtt_username, self.mqtt_password)
        # self.client.on_connect = self.__on_mqtt_connect

    def wav_filename(self):
        return self.audio_dir + 'chunk-%d.wav' % current_milli_time(0)

    def open_wave_file(self, path, sample_rate):
        """Opens a .wav file.
        Takes path, number of channels and sample rate.
        """
        wf = wave.open(path, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        return wf

    def asrmon_filename(self, suffix):
        return self.audio_dir + 'asrmon-%d.wav' % suffix

    def open_asrmon_file(self, path):
        """Opens a .wav file.
        Takes path, number of channels and sample rate.
        """
        am = wave.open(path, 'wb')
        am.setnchannels(1)
        am.setsampwidth(2)
        am.setframerate(self.asr_sample_rate)
        return am

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
            self.transcription_file.writerow(trans)


    def transcribe(self):
        """
        Monitor transcription queue for incoming audio to be transcribed either with local or remote Whisper and add
        resulting transcriptions to result list
        :return:
        """

        while self.is_running or not self.transcription_queue.empty():
            # this call blocks until an element can be retrieved from queue
            audio_segment, start, end = self.transcription_queue.get()
            conv_params = self.config['whisper_transcription']
            try:
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
                       'start':start, 'end':end }
                self.send_transcription(res)
            except:
                del self.whisper_model.model
                del self.whisper_model
                self.__init_whisper()
        logger.info("Leaving transcribe")


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
                frame = np.frombuffer(audio, dtype=np.int16)
                # monitor what comes in
                if self.am:
                    self.am.writeframes(audio)
                # data will be self.sample_rate, mono, np.int16 ndarray
                frame = self.resample(frame, self.channels, self.sample_rate)
                #print('d', np.shape(data))
                #print('vb1', len(voice_buffers))
                voice_buffers.extend(frame.tolist())
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
                    if "monitor_asr" in self.config:
                        self.wf = self.open_wave_file(
                        self.asrmon_filename(is_voice), self.asr_sample_rate)
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


    async def run_micro(self):
        self.loop = asyncio.get_running_loop()
        cb = lambda inp, frames: self.callback(inp, frames, None, None)
        pipeline = self.config["pipeline"] if "pipeline" in self.config \
            else gm.PIPELINE
        try:
            self.device = gm.GstreamerMicroSink(callback=cb, pipeline_spec=pipeline)
            self.device.start()
            if "monitor_mic" in self.config:
                self.am = self.open_wave_file(self.wav_filename(),
                                              self.sample_rate)

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

    async def read_file(self, wf):
        buffer_size = int(self.sample_rate * 0.2) # 0.2 seconds of audio
        while self.is_running:
            data = wf.readframes(buffer_size)
            if len(data) == 0:
                self.is_running = False
                await self.audio_queue.put(self.silence_buffer)
            else:
                await self.audio_queue.put(data)

    async def read_file_and_monitor(self, wf):
        with self.open_wave_file(self.wav_filename(), self.sample_rate) \
            as self.am:
            await self.read_file(wf)

    async def run_file(self, wf):
        self.loop = asyncio.get_running_loop()
        processing = asyncio.create_task(self.audio_loop())
        reading = asyncio.create_task(self.read_file(wf))
        await reading
        await processing
        logger.info("Leaving run_file")
        self.transcribe_thread.join()

async def process_files(config_file, files, output_dir):
    config = load_config(config_file);
    for file in files:
        filename = os.path.splitext(os.path.basename(file))[0]
        root = output_dir if output_dir else os.path.dirname(file)
        if root[:-1] != os.sep:
            root += os.sep
        outdir = root + filename
        # + os.sep + "chunks"
        if not os.path.exists(outdir):
            logger.info("Creating {}".format(outdir))
            os.makedirs(outdir)
        with wave.open(file, "rb") as wf:
            with open(outdir + os.sep + filename + '.csv', 'w') as csvfile:
                fieldnames = ['start', 'end', 'text']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                channels = wf.getnchannels()
                sample_rate = wf.getframerate()
                buffer_size = int(sample_rate * 0.2) # 0.2 seconds of audio
                config['monitor_mic'] = True
                config['monitor_asr'] = True
                config['sample_rate'] = sample_rate
                config['channels'] = channels
                config['audio_dir'] = outdir + os.sep
                ms = WhisperMicroServer(config, micro=False, transcription_file=writer)
                await ms.run_file(wf)
    logger.info("Leave process_files")




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
    except:
        ms.stop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
         prog='Whisper Server',
         description='Listen to microphone and transcribe input or analyse a set of files',
         epilog='')
    parser.add_argument("-c", "--config", metavar='config', type=str, required=True, help='config file')
    parser.add_argument("-o", "--output-dir", metavar='output_dir', type=str, required=False, help='output directory for chunks')
    parser.add_argument('files', metavar='files', type=str, nargs='*')
    args = parser.parse_args()

    if (args.files):
        asyncio.run(process_files(args.config, args.files, args.output_dir))
    else:
        main(args.config)
