import os
import time
import socket

from compress import run_coder, run_decoder
from constants.constant import DIR_NAME, DIR_PATH_INPUT, DIR_PATH_OUTPUT, is_quantize, save_rescaled_out
from core import load_and_rescaled
from common.logging_sd import configure_logger
import cv2

logger = configure_logger(__name__)

logger.debug(f"compressing files for is_quantize = {str(is_quantize)}")

if not os.path.exists(DIR_PATH_INPUT):
    os.makedirs(DIR_PATH_INPUT)
if not os.path.exists(DIR_PATH_OUTPUT):
    os.makedirs(DIR_PATH_OUTPUT)

logger.debug(f"get files in dir = {DIR_NAME}")

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('localhost', 9090))

while True:
    for rescaled_img, image, img_name, save_parent_dir_name, save_dir_name in load_and_rescaled():
        # функции НС
        compress_img = run_coder(cv2.cvtColor(rescaled_img, cv2.COLOR_BGR2RGB))
        sock.sendall(compress_img)
        data = sock.recv(1024)  # получаем данные с сервера
        print("Server sent: ", data.decode())

print('Close')
sock.close()
