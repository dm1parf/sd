import os
from multiprocessing import Queue
from constants.constant import DIR_PATH_INPUT, DIR_PATH_OUTPUT, is_save
from utils import save_img, create_dir
from core import latent_to_img
import cv2
import socket


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

queue_img = Queue()

while True:
    #TODO основной поток
    compress_img = con.recv(30000)           # получаем данные от клиента
    queue_img.put(compress_img)

    dir_name = count
    if not os.path.exists(f"{DIR_PATH_OUTPUT}/{dir_name}_run"):
        create_dir(DIR_PATH_OUTPUT, f"{dir_name}_run")
    save_parent_dir_name = f"{dir_name}_run"

    #TODO дополнительный поток
    uncompress_img = latent_to_img(queue_img.get())

    if is_save:
        save_img(uncompress_img, path=f"{save_parent_dir_name}", name_img=f'image{count}.jpg')

    cv2.imshow(window_name, uncompress_img)
    cv2.waitKey(25)
    count = count + 1
    message = '200'
    con.send(message.encode())
    print(count)

con.close()  # закрываем подключение
