import os
import cv2

dataset_directory = "dataset"
dest_frames = 1000
basic_size = (1280, 720)
float_mode = True

bad_files = [".DS_Store"]
image_formats = [".jpg", ".png"]
video_formats = [".mp4", ".mov"]
test_mode = True


def crop_video(video_fn, dest_dir, dframes, bsize):
    global float_mode

    os.makedirs(dest_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_fn)
    length = int(cv2.VideoCapture.get(cap, cv2.CAP_PROP_FRAME_COUNT))

    if length < dframes:
        dframes = length

    if float_mode:
        every_frame = length / dframes
        starter = ((length-1) - (dframes-1)*every_frame)/2
        count_positions = [int(starter + i*every_frame) for i in range(dframes)]
    else:
        every_frame = length // dframes
        starter = ((length-1) - (dframes-1)*every_frame)//2
        count_positions = [starter + i*every_frame for i in range(dframes)]

    for i, position in enumerate(count_positions):
        next_path = os.path.join(dest_dir, str(i) + ".jpg")
        cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        ret, frame = cap.read()

        if not ret:
            position1 = position
            position2 = position
            while True:
                position1 -= 1
                position2 += 1

                cap.set(cv2.CAP_PROP_POS_FRAMES, position1)
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, position2)
                    ret, frame = cap.read()

                    if not ret:
                        continue
                break

        frame = cv2.resize(frame, bsize, interpolation=cv2.INTER_AREA)
        cv2.imwrite(next_path, frame)

    cap.release()


def resize_image(image_fn, bsize):
    image = cv2.imread(image_fn)
    height, width, _ = image.shape

    if (height == bsize[0]) and (width == bsize[0]):
        return

    if (height > bsize[0]) and (width > bsize[1]):
        interpolation_mode = cv2.INTER_AREA
    else:
        interpolation_mode = cv2.INTER_CUBIC
    image = cv2.resize(image, bsize, interpolation=interpolation_mode)

    cv2.imwrite(image_fn, image)


for (root, dirs, filenames) in os.walk(dataset_directory):
    # Обработка злых файлов
    for bad_file in bad_files:
        if bad_file in filenames:
            bad_filepath = os.path.join(root, bad_file)

            print(bad_filepath)

            os.remove(bad_filepath)
            filenames.remove(bad_file)

    # Коррекция файлов
    for filename in filenames:
        filepath = os.path.join(root, filename)

        print(filepath)

        base, ext = os.path.splitext(filename)
        ext = ext.lower()

        if ext in image_formats:
            # Коррекция изображений
            resize_image(filepath, basic_size)
        elif ext in video_formats:
            # Коррекция видео
            crop_dir = os.path.join(root, base)
            crop_video(filepath, crop_dir, dest_frames, basic_size)
            os.remove(filepath)
        else:
            # Коррекция файлов непредвиденных форматов
            if test_mode:
                print("!!! WRONG EXTENSION !!!")
                print(filepath)
                break
            else:
                os.remove(filepath)

