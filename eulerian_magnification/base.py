import cv2
import numpy as np
import scipy.fftpack
# import scipy.signal
from scipy import signal
from matplotlib import pyplot

from eulerian_magnification.io import play_vid_data
from eulerian_magnification.pyramid import create_laplacian_video_pyramid, collapse_laplacian_video_pyramid
from eulerian_magnification.transforms import temporal_bandpass_filter


def eulerian_magnification(vid_data, fps, freq_min, freq_max, amplification, pyramid_levels=4, skip_levels_at_top=2):
    vid_pyramid = create_laplacian_video_pyramid(vid_data, pyramid_levels=pyramid_levels)
    for i, vid in enumerate(vid_pyramid):
        if i < skip_levels_at_top or i >= len(vid_pyramid) - 1:
            continue

        bandpassed = temporal_bandpass_filter(vid, fps, freq_min=freq_min, freq_max=freq_max, amplification_factor=amplification)

        play_vid_data(bandpassed)

        vid_pyramid[i] += bandpassed
        play_vid_data(vid_pyramid[i])

    vid_data = collapse_laplacian_video_pyramid(vid_pyramid)
    return vid_data



    # def eulerian_magnification_live(frame, fps, freq_min, freq_max, amplification, pyramid_levels=4, skip_levels_at_top=2):
    #     vid_pyramid = create_laplacian_image_pyramid(vid_data, pyramid_levels=pyramid_levels)
    #     for i, vid in enumerate(vid_pyramid):
    #         if i < skip_levels_at_top or i >= len(vid_pyramid) - 1:
    #             # ignore the top and bottom of the pyramid. One end has too much noise and the other end is the
    #             # gaussian representation
    #             continue

    #         bandpassed = temporal_bandpass_filter(vid, fps, freq_min=freq_min, freq_max=freq_max, amplification_factor=amplification)

    #         play_vid_data(bandpassed)

    #         vid_pyramid[i] += bandpassed
    #         play_vid_data(vid_pyramid[i])

    #     vid_data = collapse_laplacian_video_pyramid(vid_pyramid)
    #     return vid_data


def show_frequencies(vid_data, fps, bounds=None):
    """Graph the average value of the video as well as the frequency strength"""
    averages = []

    if bounds:
        for x in range(1, vid_data.shape[0] - 1):
            averages.append(vid_data[x, bounds[2]:bounds[3], bounds[0]:bounds[1], :].sum())
    else:
        for x in range(1, vid_data.shape[0] - 1):
            averages.append(vid_data[x, :, :, :].sum())

    averages = averages - min(averages)

    charts_x = 1
    charts_y = 2
    pyplot.figure(figsize=(20, 10))
    pyplot.subplots_adjust(hspace=.7)

    pyplot.subplot(charts_y, charts_x, 1)
    pyplot.title("Pixel Average")
    pyplot.xlabel("Time")
    pyplot.ylabel("Brightness")
    pyplot.plot(averages)

    freqs = scipy.fftpack.fftfreq(len(averages), d=1.0 / fps)
    fft = abs(scipy.fftpack.fft(averages))
    idx = np.argsort(freqs)

    pyplot.subplot(charts_y, charts_x, 2)
    pyplot.title("FFT")
    pyplot.xlabel("Freq (Hz)")
    freqs = freqs[idx]
    fft = fft[idx]

    freqs = freqs[int(len(freqs) / 2) + 1:]
    fft = fft[int(len(fft) / 2) + 1:]
    pyplot.plot(freqs, abs(fft))
    pyplot.savefig('graph_doll.png')
    print("pyplot show")
    pyplot.show()

    # fs = 10e3
    # N = 1e5
    # amp = 2 * np.sqrt(2)
    # noise_power = 0.01 * fs / 2
    # time = np.arange(N) / float(fs)
    # mod = 500*np.cos(2*np.pi*0.25*time)
    # carrier = amp * np.sin(2*np.pi*3e3*time + mod)
    # noise = np.random.normal(scale=np.sqrt(noise_power),
    #                          size=time.shape)
    # noise *= np.exp(-time/5)
    # x = carrier + noise

    # f, t, Zxx = signal.stft(averages, fs, nperseg=1000)
    # pyplot.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp)
    # pyplot.title('STFT Magnitude')
    # pyplot.ylabel('Frequency [Hz]')
    # pyplot.xlabel('Time [sec]')
    # pyplot.show()



def gaussian_video(video, shrink_multiple):
    """Create a gaussian representation of a video"""
    vid_data = None
    for x in range(0, video.shape[0]):
        frame = video[x]
        gauss_copy = np.ndarray(shape=frame.shape, dtype="float")
        gauss_copy[:] = frame
        for i in range(shrink_multiple):
            gauss_copy = cv2.pyrDown(gauss_copy)

        if x == 0:
            vid_data = np.zeros((video.shape[0], gauss_copy.shape[0], gauss_copy.shape[1], 3))
        vid_data[x] = gauss_copy
    return vid_data


def laplacian_video(video, shrink_multiple):
    vid_data = None
    frame_count, height, width, colors = video.shape

    for i, frame in enumerate(video):
        gauss_copy = np.ndarray(shape=frame.shape, dtype="float")
        gauss_copy[:] = frame

        for _ in range(shrink_multiple):
            prev_copy = gauss_copy[:]
            gauss_copy = cv2.pyrDown(gauss_copy)

        laplacian = prev_copy - cv2.pyrUp(gauss_copy)

        if vid_data is None:
            vid_data = np.zeros((frame_count, laplacian.shape[0], laplacian.shape[1], 3))
        vid_data[i] = laplacian
    return vid_data


def combine_pyramid_and_save(g_video, orig_video, enlarge_multiple, fps, save_filename='media/output.avi'):
    """Combine a gaussian video representation with the original and save to file"""
    width, height = get_frame_dimensions(orig_video[0])
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    print("Outputting to %s" % save_filename)
    writer = cv2.VideoWriter(save_filename, fourcc, fps, (width, height), 1)
    for x in range(0, g_video.shape[0]):
        img = np.ndarray(shape=g_video[x].shape, dtype='float')
        img[:] = g_video[x]
        for i in range(enlarge_multiple):
            img = cv2.pyrUp(img)

        img[:height, :width] = img[:height, :width] + orig_video[x]
        res = cv2.convertScaleAbs(img[:height, :width])
        writer.write(res)


def get_frame_dimensions(frame):
    """Get the dimensions of a single frame"""
    height, width = frame.shape[:2]
    return width, height


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data, axis=0)
    return y
