#!/usr/bin/env python3

import sys
from pathlib import Path
import logging
import numpy as np
import traceback
import requests

from transcriptor import Transkriptor, modroot, main, named_tuple_to_dictionary

from faster_whisper import WhisperModel

# configure logger
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class WhisperMicroServer(Transkriptor):

    def __init__(self, config, transcription_file=None):
        super().__init__(config, "whisperasr", transcription_file)
        self.topics.update({
            self.pid + '/set_prompt': self._on_prompt_msg,
        })
        self.__init_recognizer()

    def __init_recognizer(self):
        self.whisper: WhisperModel
        if 'whisper' not in self.config:
            logger.error('no whisper config section: minimally specify model size or URL')
            sys.exit(1)
        whisper_config = self.config['whisper']
        self.whisper_url = whisper_config.get('url', None)
        if self.whisper_url:
            return   # using remote ASR
        if 'device' not in whisper_config or whisper_config['device'] == 'cpu':
            whisper_config['device'] = 'cpu'
            if 'compute_type' not in whisper_config:
                whisper_config['compute_type'] = 'int16'
        else:
            whisper_config['device'] = 'cuda'
            if 'compute_type' not in whisper_config:
                whisper_config['compute_type'] = 'float32'
        if 'whisper_transcription' not in self.config:
            self.config['whisper_transcription'] = dict()
        if self.language and 'language' not in self.config['whisper_transcription']:
            self.config['whisper_transcription']['language'] = self.language
        logger.info(
            f"initializing {whisper_config['model_size']} model "
            f"for {whisper_config['device']} {whisper_config['compute_type']} ...")
        model_path = modroot / 'whisper' / whisper_config['model_size']
        self.whisper = WhisperModel(str(model_path),
                                    device=whisper_config['device'],
                                    compute_type=whisper_config['compute_type'])
        logger.info("Whisper model initialized")


    def transcribe(self, audio_segment, start, end):
        conv_params = self.config['whisper_transcription']
        # if 'initial_prompt' not in conv_params and self.prompt:
        if self.initial_prompt:
            conv_params['initial_prompt'] = self.initial_prompt
        result = None
        if not self.whisper_url:
            try:
                logger.info("transcribing")
                segments, info = self.whisper.transcribe(np.array(audio_segment), **conv_params)
                logger.info("done ...")
                transcripts = []
                for segment in segments:
                    conv_dict = named_tuple_to_dictionary(segment)
                    transcripts.append(conv_dict)
                result = {'info': named_tuple_to_dictionary(info),
                          'segments': transcripts,
                          'start': start, 'end': end,
                          'source': self.audio_source}
            except Exception as ex:
                logger.error('whisper exception: {}'.format(ex))
                traceback.print_exc()
                del self.whisper.model
                del self.whisper
                self.__init_recognizer()
        else:
            segment = np.array(audio_segment, dtype=np.float32)
            segment /= 32768
            response = requests.post(url=self.whisper_url,
                                     data=segment.tobytes(),
                                     headers={'Content-Type': 'application/octet-stream'},
                                     params=conv_params)
            try:
                result = response.json()
                result['start'] = start
                result['end'] = end
                result['source'] = self.audio_source
            except Exception as ex:
                logger.error(f'wrong response: {response} {ex}')

        if result and 'segments' in result:
            text = ' '.join(map(lambda x: x['text'], result['segments']))
            result['text'] = text
            self.transcribe_success(result, audio_segment)

    def start_buffer(self, buffer):
        return super().start_buffer(buffer)

    def continue_buffer(self, buffer):
        return super().continue_buffer(buffer)

    def end_buffer(self, buffer, end_time):
        self.transcription_queue.put((self.vad_state.last_segment(),
                                      self.start_time, end_time))
        return super().end_buffer(buffer, end_time)


if __name__ == '__main__':
    # one level up for src/
    modroot = Path(sys.argv[0]).parent.parent.absolute() / 'models'
    main(WhisperMicroServer, "Whisper Server")
