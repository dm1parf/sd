import os
import threading
from multiprocessing import Queue
from constants.constant import DIR_PATH_INPUT, DIR_PATH_OUTPUT, is_save
from utils import save_img, create_dir
from core import latent_to_img
import cv2
import socket
import concurrent.futures


def uncompress(img):
    latent_to_img(img)


def worker():
    while True:
        item = queue_of_futures.get()
        while item.running():
            pass

        dir_name = count
        if not os.path.exists(f"{DIR_PATH_OUTPUT}/{dir_name}_run"):
            create_dir(DIR_PATH_OUTPUT, f"{dir_name}_run")
        save_parent_dir_name = f"{dir_name}_run"

        if is_save:
            save_img(item, path=f"{save_parent_dir_name}", name_img=f'image{count}.jpg')

        cv2.imshow(window_name, item)
        cv2.waitKey(25)
        count = count + 1
        message = '200'
        con.send(message.encode())

        print(count)

        queue_of_futures.task_done()

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

queue_of_futures = Queue()
with concurrent.futures.ThreadPoolExecutor() as executor:
    threading.Thread(target=worker, daemon=True).start()
    while True:

        compress_img = con.recv(30000)  # получаем данные от клиента
        queue_of_futures.put(executor.submit(uncompress, compress_img))

    queue_of_futures.join()
    con.close()  # закрываем подключение
