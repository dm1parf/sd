import cv2

this_video = "output_8x_2.mp4"
dest_video = "8x.avi"
basic_size = (1280, 720)
dest_fps = 120


cap = cv2.VideoCapture(this_video)
source_fps = int(cv2.VideoCapture.get(cap, cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(dest_video, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), dest_fps, basic_size)

frame_num = -1
while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, basic_size, interpolation=cv2.INTER_AREA)
    out.write(frame)

    frame_num += 1
    print("> Кадр {}".format(frame_num))

out.release()
cap.release()
