import concurrent.futures
import os
import queue
import socket
import threading
import time

import cv2

from common.logging_sd import configure_logger
from compress import createSd
from constants.constant import DIR_PATH_INPUT, DIR_PATH_OUTPUT, is_save, PREDICTION_MODEL_PATH, REAL, FAKE, REAL_NAME, \
    FAKE_NAME, USE_PREDICTION, Platform
from core import latent_to_img
from prediction import Model, DMVFN
from utils import save_img, create_dir


def uncompress(img):
    decoderStartTime = time.time()
    res = latent_to_img(img)
    logger.debug(f"time for clear decoder: {time.time()-decoderStartTime}")
    return res


def predict_img(list_of_images_in_futures):
    buffer = []
    for img in list_of_images_in_futures[-2:]:
        while img.running():
            pass
        buffer.append(img.result())
    predictionStartTime = time.time()
    res = model.predict(buffer)
    logger.debug(f"time for clear prediction: {time.time()-predictionStartTime}")
    return res


def worker():
    global count
    while True:
        item = queue_of_futures.get()
        while item.running():
            pass
        result_img = item.result()

        dir_name = count
        if not os.path.exists(f"{DIR_PATH_OUTPUT}/{dir_name}_run"):
            create_dir(DIR_PATH_OUTPUT, f"{dir_name}_run")
        save_parent_dir_name = f"{dir_name}_run"

        if is_save:
            save_img(result_img, path=f"{save_parent_dir_name}", name_img=f'image{count}.jpg')

        cv2.imshow(window_name, result_img)
        cv2.waitKey(25)
        count = count + 1
        message = '200'
        con.send(message.encode())

        print(count)

        queue_of_futures.task_done()


createSd(Platform.CLIENT)
logger = configure_logger(__name__)
count = 0

if not os.path.exists(DIR_PATH_INPUT):
    os.makedirs(DIR_PATH_INPUT)
if not os.path.exists(DIR_PATH_OUTPUT):
    os.makedirs(DIR_PATH_OUTPUT)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(('', 9090))
sock.listen(1)
con, _ = sock.accept()  # принимаем клиента

print('Sock name: {}'.format(sock.getsockname()))

window_name = 'Video'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

if USE_PREDICTION:
    model = Model(DMVFN(PREDICTION_MODEL_PATH))
    pattern = [REAL_NAME] * REAL + [FAKE_NAME] * FAKE
    pattern_counter = 0
    restored_imgs = []

queue_of_futures = queue.Queue()

with concurrent.futures.ThreadPoolExecutor() as uncompress_executor:
    with concurrent.futures.ThreadPoolExecutor() as predict_executor:
        threading.Thread(target=worker, daemon=True).start()
        while True:

            compress_img = con.recv(30000)  # получаем данные от клиента

            if USE_PREDICTION:
                if pattern[pattern_counter % len(pattern)] == REAL_NAME:
                    fut = uncompress_executor.submit(uncompress, compress_img)

                elif pattern[pattern_counter % len(pattern)] == FAKE_NAME:
                    fut = predict_executor.submit(predict_img, restored_imgs)

                queue_of_futures.put(fut)

                restored_imgs.append(fut)

                if len(restored_imgs) > 2:
                    del restored_imgs[0]
            else:
                queue_of_futures.put(uncompress_executor.submit(uncompress, compress_img))

    queue_of_futures.join()
    con.close()  # закрываем подключение
