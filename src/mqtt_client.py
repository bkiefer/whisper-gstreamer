import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion
import logging

logger: logging.Logger
logger = logging.getLogger(__file__)

class MqttClient():
    valid_mqtt_keys = { 'host', 'port', 'bind_address', 'bind_port',
                        'keepalive', 'clean_start',
                        'username', 'password' }

    def __init__(self, pid, config):
        self.pid = pid
        self.topics = {}
        self.mqtt_config = { 'host': 'localhost' }
        self.mqtt_config.update(config)
        for key in self.mqtt_config:
            if key not in self.__class__.valid_mqtt_keys:
                del(key, self.mqtt_config)
        self.client: mqtt.Client
        self.client = mqtt.Client(CallbackAPIVersion.VERSION2)
        if 'username' in self.mqtt_config and 'password' in self.mqtt_config:
            self.client.username_pw_set(self.mqtt_config['username'],
                                        self.mqtt_config['password'])
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

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
        logger.debug(f"Received message {str(message.payload)} "
                     + "on topic {message.topic} with QoS {str(message.qos)}")
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

    def mqtt_connect(self, forever = False):
        self.client.connect(**self.mqtt_config)
        if forever:
            self.client.loop_forever()
        else:
            self.client.loop_start()

    def mqtt_disconnect(self):
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
