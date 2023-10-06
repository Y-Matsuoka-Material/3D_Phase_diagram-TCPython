# coding: UTF-8
import cv2 as cv
import os
import subprocess
import atexit

class VideoWriter:
    def __init__(self, filename, fps, size, encode=True, delete_original=True):
        self.filename = filename.rsplit(".", 1)[0] + ".avi"
        self.fourcc = cv.VideoWriter_fourcc(*'XVID')
        self.fps = fps
        self.size = size
        self.encode = encode
        self.delete_original = delete_original
        self.ffmpeg_available = self.is_ffmpeg_available()
        self.alive = True
        if self.ffmpeg_available and self.encode:
            self.fourcc = cv.VideoWriter_fourcc(*'I420')
        self.video_writer = cv.VideoWriter(self.filename, self.fourcc, self.fps, self.size)
        atexit.register(self.finalize)

    def __del__(self):
        self.finalize()

    def write(self, frame):
        self.video_writer.write(frame)

    def finalize(self):
        if self.alive:
            self.alive = False
            self.video_writer.release()
            if self.ffmpeg_available and self.encode:
                self.encode_and_delete(self.filename, self.delete_original)

    @classmethod
    def is_ffmpeg_available(self):
        paths_win = os.environ["PATH"].split(";")
        winflag = any([os.path.isfile(f"{p}\\ffmpeg.exe") for p in paths_win])
        paths_lin = os.environ["PATH"].split(":")
        linflag = any([os.path.isfile(f"{p}/ffmpeg") for p in paths_lin])
        return winflag or linflag

    @classmethod
    def encode_and_delete(self, source_file, delete_original=True):
        if subprocess.Popen is not None:
            enc = subprocess.Popen(["ffmpeg", "-y", "-i", source_file, source_file.replace(".avi", ".mp4")],
                                   stdin=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL)
            if enc.wait() == 0 and delete_original:
                os.remove(source_file)