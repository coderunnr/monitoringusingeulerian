import os
import requests
import eulerian_magnification as em
from eulerian_magnification.io import save_video, load_video_float

SOURCE_VIDEOS_DIR = 'eulerian_source_videos'
DESTINATION_VIDEOS_DIR = 'showcase'

def apply_eulerian():
    image_processing = 'laplacian'
    pyramid_levels = 4

    name='output_live'
    amplification_factor=10
    cutoff=16
    lower_hertz=0.4
    upper_hertz=0.8
    framerate=30

    source_filename = name + '.avi'

    source_path = os.path.join(SOURCE_VIDEOS_DIR, source_filename)
    vid, fps = load_video_float(source_path)
    vid = em.eulerian_magnification(
        vid, fps,
        freq_min=lower_hertz,
        freq_max=upper_hertz,
        amplification=amplification_factor,
        pyramid_levels=pyramid_levels
    )
    source_path = os.path.join(DESTINATION_VIDEOS_DIR, source_filename)
    file_name = os.path.splitext(source_path)[0]
    file_name = file_name + "__" + image_processing + "_levels" + str(pyramid_levels) + "_min" + str(
        lower_hertz) + "_max" + str(upper_hertz) + "_amp" + str(amplification_factor)
    save_video(vid, fps, file_name + '.avi')


if __name__ == '__main__':
    apply_eulerian()