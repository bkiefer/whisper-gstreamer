#!/usr/bin/env python3

import sys
from pathlib import Path
import logging
import json

from transcriptor import Transkriptor, modroot, main, current_milli_time
from vosk import Model, KaldiRecognizer, SetLogLevel

# configure logger
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class VoskMicroServer(Transkriptor):

    def __init__(self, config, transcription_file=None):
        super().__init__(config, "voskasr", transcription_file)
        self.silence_buffer = bytearray(self.asr_sample_rate * 6)
        self.__init_recognizer()

    def __init_recognizer(self):
        if 'vosk' not in self.config:
            logger.error('no vosk config section: minimally specify model name or path')
            sys.exit(1)
        vosk_config = self.config['vosk']
        model_path = Path(vosk_config['model_path'])
        if not model_path.is_absolute():
            model_path = modroot / 'kaldi_models' / model_path
        self.asr_model = Model(lang=self.language,
                               model_path=str(model_path.absolute()))
        self.recognizer = KaldiRecognizer(self.asr_model, self.asr_sample_rate)
        self.recognizer.SetMaxAlternatives(1)
        self.recognizer.SetWords(True)
        logger.info("Vosk model initialized")

    bullshit = { '' , 'einen', 'bin', 'the' }

    # digest audio and check if there is a result from the ASR
    def send_frames(self, buf, voice_end=current_milli_time()):
        if self.recognizer.AcceptWaveform(buf):
            data = json.loads(self.recognizer.Result())
            # PRELIMINARY SOLUTION FOR MULTIPLE ALTERNATIVES
            if data and 'alternatives' in data:
                data = data['alternatives'][0]
            if data and 'text' in data:
                text = data['text']
                if text not in VoskMicroServer.bullshit:
                    data['start'] = self.start_time
                    data['end'] = voice_end
                    data['source'] = self.audio_source
                    print(data)
                    self.transcribe_success(data, self.vad_state.last_segment())
        else:
            # partial result in self.recognizer.PartialResult()
            logger.debug(f'{self.recognizer.PartialResult()}')
        return buf

    def transcribe(self, audio_segment, start, end):
        self.start_time = start
        buf = self.intlist2bytes(audio_segment)
        self.send_frames(buf, end)
        self.send_frames(self.intlist2bytes(self.silence_buffer))
        self.start_time = None

    def start_buffer(self, buffer):
        return self.send_frames(super().start_buffer(buffer))

    def continue_buffer(self, buffer):
        return self.send_frames(super().continue_buffer(buffer))

    def end_buffer(self, buffer, end_time):
        buf = self.write_buffer(buffer)
        self.send_frames(buf, end_time)
        self.send_frames(self.write_buffer(self.silence_buffer))
        if self.wf:
            self.wf.close()
            self.wf = None
        self.start_time = None
        return buf

if __name__ == '__main__':
    # one level up for src/
    modroot = Path(sys.argv[0]).parent.parent.absolute() / 'models'
    main(VoskMicroServer, "Vosk ASR")
