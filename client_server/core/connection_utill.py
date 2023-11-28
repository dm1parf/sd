import socket
import time

from common.logging_sd import configure_logger

logger = configure_logger(__name__)


def create_client(url, port, function, params):
    is_connect = False
    con = None

    while not is_connect:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind((url, port))
            sock.listen(1)
            logger.info('create socket is OK')
            con, _ = sock.accept()  # принимаем клиента
            logger.info('Sock name: {}'.format(sock.getsockname()))
            is_connect = True

            function(con, params)
        except Exception as err:
            logger.debug(f'connection to host {url}:{port} exception; Reason: {err}')
            is_connect = False
            time.sleep(10)
        finally:
            if con is not None:
                con.close()  # закрываем подключение


def create_server(url, port, function, param):
    is_connect = False
    sock = None
    while not is_connect:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((url, port))
            is_connect = True
            function(sock, param)
        except Exception as ex:
            is_connect = False
            logger.debug(f'connection to host {url}:{port} exception; Reason: {ex}')
            time.sleep(10)
        finally:
            if sock is not None:
                sock.close()
