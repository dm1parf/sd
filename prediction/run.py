import sys
from common.logging import configure_logger
from common.utils import check_args, create_data_dirs, get_imgs, parse_args
from metrics.metrics import run_metrics
logger = configure_logger(__name__)


def main(args):

    exp_dir = create_data_dirs(args)

    if not check_args(args):
        sys.exit('Incorrect args!\n Exiting...')

    images = get_imgs(args)

    # images = scale_imgs(images)

    # result_image = model(image1, image2)

    for index, image_pair in enumerate(images):
        run_metrics(*image_pair, exp_dir, index)


if __name__ == '__main__':
    logger.info('Starting...')

    args = parse_args()
    main(args)
