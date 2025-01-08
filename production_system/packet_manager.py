import struct


class PacketManager:
    supported_versions = (0,)

    def __init__(self, version):
        assert version in self.supported_versions, NotImplementedError("Wrong version:", version)

        self.version = version

        self.message_handlers = {
            5: self.parse_vstr,
        }

    def parse_packet(self, packet):
        """Разобрать пакет FPV-CTVP."""

        (version, stream_identifier, segment_type,
         message_type, message_length) = struct.unpack('>BHBBQ', packet[:13])
        message = packet[13:13+message_length]

        message_data = self.parse_message(message_type, message)

        packet_data = {
            "msg_data": message_data,
            "version": version,
            "stream_id": stream_identifier,
            "segment_type": segment_type,
            "message_type": message_type
        }

        return packet_data

    def parse_message(self, message_type, message):
        """Разобрать сообщение FPV-CTVP."""

        message_handler = self.message_handlers.get(message_type, None)

        if message_handler is None:
            raise NotImplementedError("Wrong message type:", message_type)
        else:
            message_content = message_handler(message)
            return message_content

    def parse_vstr(self, message):
        """Разобрать сообщение VSTR FPV-CTVP."""

        (frame_num, segment_num, total_segments,
         height, width, cfg_num, encryption_num,
         payload_length) = struct.unpack(">QHHIIBBQ", message[:30])
        payload = message[30:]

        message_data = {
            "frame_num": frame_num,
            "seq_num": segment_num,
            "total_seq": total_segments,
            "payload": payload,
            "height": height,
            "width": width,
            "cfg_num": cfg_num,
            "encryption_num": encryption_num
        }

        return message_data

    def pack_packet(self, message, message_type):
        """Запаковать пакет FPV-CTVP."""

        version = 0
        stream_identifier = 0
        segment_type = 0
        message_length = len(message)

        packet = struct.pack(">BHBBQ", version,
                             stream_identifier,
                             segment_type,
                             message_type,
                             message_length)
        packet += message

        return packet

    def pack_vstr(self, frame_num, height, width, segment_num, total_segments, cfg_num, payload):
        """Запаковать сообщение FPV-CTVP."""

        encryption_num = 0
        payload_length = len(payload)

        message = struct.pack(">QHHIIBBQ", frame_num,
                              segment_num,
                              total_segments,
                              height,
                              width,
                              cfg_num,
                              encryption_num,
                              payload_length)
        message += payload

        return message
