import os
import eulerian_magnification as em
from eulerian_magnification.io import save_video, load_video_float

SOURCE_VIDEOS_DIR = 'showcase'
name='output_live_project_demo'
source_filename = name + '.avi'
source_path = os.path.join(SOURCE_VIDEOS_DIR, source_filename)
vid, fps = load_video_float(source_path)
em.show_frequencies(vid,fps)