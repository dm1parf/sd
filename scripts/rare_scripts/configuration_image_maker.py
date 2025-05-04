import os
import cv2
from scripts.stand1.stand1_decoder import ConfigurationGuardian


input_file = "57.png"
out_dir = "image_maker"

guard = ConfigurationGuardian()
os.makedirs(out_dir, exist_ok=True)
all_configurations = [1, 2, 3, 4, 5]
source_frame = cv2.imread(input_file)
pic_height, pic_width = source_frame.shape[:2]


for cfg in all_configurations:
    neuro_codec = guard.get_configuration(cfg)

    latent = neuro_codec.encode_frame(source_frame)
    dest_frame = neuro_codec.decode_frame(latent, dest_width=pic_width, dest_height=pic_height)

    dest_file = os.path.join(out_dir, str(cfg) + ".jpg")
    cv2.imwrite(dest_file, dest_frame)
