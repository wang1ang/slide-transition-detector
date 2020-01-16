from shotdetector import *
import cv2
import numpy as np
from matplotlib import pyplot as plt
class BorderDetector(FrameCallback):
    def __init__(self):
        super(BorderDetector, self).__init__()
    def OnFrame(self, id, time, frame):
        frame = np.array(frame, dtype=np.integer)
        dx = np.max(np.abs(np.diff(frame, axis = 1)), axis=2)
        dy = np.max(np.abs(np.diff(frame, axis = 0)), axis=2)
        row_diff = np.sum(np.abs(np.diff(dx, axis=0)), axis=1) / dx.shape[1]
        col_diff = np.sum(np.abs(np.diff(dy, axis=1)), axis=0) / dy.shape[0]
        top = next(i for i, x in enumerate(np.append(row_diff, 1)) if x >= 1)
        bottom = next(i for i, x in enumerate(np.append(row_diff[::-1], 1)) if x >= 1)
        left = next(i for i, x in enumerate(np.append(col_diff, 1)) if x >= 1)
        right = next(i for i, x in enumerate(np.append(col_diff[::-1], 1)) if x >= 1)
        self.features.append([top, bottom, left, right])
        return self.features[-1]
    def Finalize(self):
        pass

class SlideDetector(FrameCallback):
    def __init__(self):
        super(SlideDetector, self).__init__()
        self.border = BorderDetector()
        self.AddCallback(self.border)
    def OnFrame(self, id, time, frame):
        if type(frame) is Shot:
            self.OnShot(frame)
    def OnShot(self, shot):
        score = 0
        score += 1 if shot.stopTime - shot.startTime > 1 else 0
        score += 1 if shot.stable < 1 else 0
        score += -2 if shot.stable > 2 else 0
        border_top, border_bottom, border_left, border_right = self.border.OnFrame(shot.id, shot.time, shot.frame)
        score += 1 if border_top > 5 else 0
        score += 1 if border_bottom > 5 else 0
        score += 1 if border_left > 5 else 0
        score += 1 if border_right > 5 else 0
        #score += 1 if face_count == 0
        #score += 1 if tag.Length > 0;

        isslide = score >= 5
        if isslide:
            for c in self.callbacks:
                c.OnFrame(shot.id, shot.time, shot.frame)
        self.features.append([isslide])
        return isslide

    def Finalize(self):
        pass

import argparse
if __name__ == "__main__":
    Parser = argparse.ArgumentParser(description="Slide Detector")
    Parser.add_argument("-i", "--input", help="video device number or path to video file")
    Parser.add_argument("-o", "--outpath", help="path to output video file", default="slides/", nargs='?')
    Parser.add_argument("-f", "--fileformat", help="file format of the output images e.g. '.jpg'",
                        default=".jpg", nargs='?')
    Args = Parser.parse_args()

    parser = VideoParser(Args.input)
    
    sampler = FrameSampler(5)

    segmentor = ShotDetector()


    parser.AddCallback(sampler)
    sampler.AddCallback(segmentor)

    dumper = FrameDumper(Args.outpath, Args.fileformat)
    segmentor.AddCallback(dumper)

    #border = BorderDetector()
    #segmentor.AddCallback(border)

    slide = SlideDetector()
    segmentor.AddCallback(slide)

    dumper_slide = FrameDumper(os.path.join(Args.outpath, 'slides'), Args.fileformat)
    slide.AddCallback(dumper_slide)

    parser.Run()

    for i in range(len(segmentor.features)):
        print (segmentor.GetFeatures(i))