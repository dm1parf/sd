import socket
import socketserver

from common.logging_sd import configure_logger

logger = configure_logger(__name__)


def send_message(host, port, message):
    try:
        with socket.create_connection((host, port)) as client_socket:
            logger.debug(f"sending message from server {host}:{port}")
            client_socket.sendall(message)
    except Exception as ex:
        logger.debug(f'connection to host {host}:{port} exception; Reason: {ex}')


def create_server(host, port, handler):
    try:
        with socketserver.TCPServer((host, port), handler) as server:
            logger.debug(f"Server listening on {host}:{port}")
            server.serve_forever()
    except Exception as ex:
        logger.debug(f'connection to host {host}:{port} exception; Reason: {ex}')
        create_server(host, port, handler)
