
import cv2
class FrameCallback(object):
    def __init__(self):
        self.callbacks = []
        self.id = 0
    def AddCallback(self, callback):
        self.callbacks.append(callback)

import ui
class VideoParser(FrameCallback):
    def sanitize_device(self, device):
        try:
            return int(device)
        except (TypeError, ValueError):
            return device
    def __init__(self, device):
        super(VideoParser, self).__init__()
        self.cap = cv2.VideoCapture(self.sanitize_device(device))
        self.len = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
    def Run(self):
        progress = ui.ProgressController('Analyzing Video: ', self.len)
        progress.start()

        count = 0
        suc = True
        while suc:
            timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            suc, frame = self.cap.read()
            for c in self.callbacks:
                c.OnFrame(count, timestamp, frame)
                count += 1
                progress.update(count)
        for c in self.callbacks:
            c.Finalize()
        progress.finish()
        
class FrameSampler(FrameCallback):
    def __init__(self, fps):
        super(FrameSampler, self).__init__()
        self.fps = fps
        self.time = -1
        self.frames = 0
        self.frame = None
    def OnFrame(self, id, time, frame):
        targetTime = self.frames / self.fps
        if (abs(targetTime - time) > abs(targetTime - self.time)):
            for c in self.callbacks:
                c.OnFrame(self.frames, self.time, self.frame)
            self.frames += 1
        self.time = time
        self.frame = frame
    def Finalize(self):
        targetTime = self.frames / self.fps
        if self.time > targetTime:
            self.OnFrame(None, self.time+1/self.fps, None) # trigger previous frame
        for c in self.callbacks:
            c.Finalize()

class FrameFeat:
    def __init__(self, diff, bright0, bright1, shot, time):
        self.diff = diff
        self.bright0 = bright0
        self.bright1 = bright1
        self.shot = shot
        self.time = time
class Shot:
    def __init__(self, id, time, image=None):
        self.id = id
        self.time = time
        self.shotId = id
        self.startTime = time
        self.stopTime = time
        self.diff = 0
        self.shot = 0
        self.stable = 0
        self.image = image
import numpy as np
import math

class ShotDetector(FrameCallback):
    def __init__(self):
        super(ShotDetector, self).__init__()
        self.lastFrame = None
        self.bestFrame = None
        self.frame = None
        self.bright = 255
        self.frameFeat = []
        self.diff = []
        self.time = []
        self.shotList = []
        self.step = 4
    @staticmethod
    def get_bright(frame):
        m = np.max(frame, axis=2)
        s = np.sum(m)
        hist = cv2.calcHist([m], [0], None, [256], [0,256]).squeeze()
        pixels = m.size
        bright = 255
        acc = 0
        while acc < pixels * 0.01:
            acc += hist[bright]
            bright -= 1
        bright += 1
        return s/float(pixels), bright
        
    def run_callbacks(self):
        if self.shotList:
            shot = self.shotList[-1]
            shot.stopTime = self.frame.time
            shot.frame = self.bestFrame
            for c in self.callbacks:
                c.OnFrame(shot.id, shot.time, shot.frame) # OnShot
            shot.frame = None # useless after calling all callbacks
    def new_shot(self, id, time, frame):
        shot = Shot(id, time, frame)
        shot.shotId = id
        shot.diff = self.frameFeat[-1].diff
        shot.shot = self.frameFeat[-1].shot
        # previous frame generates the large diff
        if len(self.frameFeat) > 1:
            if self.frameFeat[-2].diff > shot.diff:
                shot.diff = self.frameFeat[-2].diff
                shot.shotId -= 1
            shot.shot = self.frameFeat[-2].shot
        # check last shot
        if self.shotList and self.shotList[-1].id == shot.id:
            prev = self.shotList[-1]
            if prev.diff < shot.diff:
                prev.diff = shot.diff
                prev.shotId = shot.shotId
        else:
            self.run_callbacks()
            self.shotList.append(shot)
            self.bestFrame = frame
    def replace_keyframe(self, id, time, frame):
        shot = self.shotList[-1]
        shot.id = id
        shot.time = time
        shot.image = frame
        self.bestFrame = frame
    def OnFrame(self, id, time, frame):
        self.frame = Shot(id, time, frame) # make these visible

        bright0, bright1 = self.get_bright(frame)
        if self.bright < bright1:
            self.bright = bright1

        if self.lastFrame is not None:
            diff = self.pixel_diff(frame, self.lastFrame)
            diff *= (255 + (255 - bright0)) / (self.bright + (255 - bright0))
            self.frameFeat.append(FrameFeat(diff, bright0, bright1, 0, time))
        else:
            self.lastFrame = frame
            self.frameFeat.append(FrameFeat(0, bright0, bright1, 0, time))
        if self.bright > bright1:
            self.bright = 0.9 * self.bright + 0.1 * bright1
        
        self.frameFeat[-1].shot = self.check_shot(id, time, frame)
        self.lastFrame = frame
    def Finalize(self):
        self.run_callbacks()
        for c in self.callbacks:
            c.Finalize()
        # copy diff
        for i in range(len(self.frameFeat)):
            self.diff[i] = self.frameFeat[i].diff
        # shot stableness
        for i in range(len(self.shotList)):
            first_id = self.shotList[i].id
            last_id = len(self.frameFeat)-1
            if i < len(self.shotList)-1: # not last
                # shrink
                last_id = self.shotList[i+1].shotId
                while first_id < last_id and self.diff[last_id - 1] < self.diff[last_id]:
                    last_id -= 1
            stable = 0
            for id in range(first_id, last_id):
                stable += self.diff[id]
            if first_id >= last_id:
                self.shotList[i].stable = self.diff[first_id]
            else:
                self.shotList[i].stable = stable / (last_id - first_id)
            self.shotList[i].stopTime = self.frameFeat[last_id].time
    def pixel_diff(self, f, g):
        return np.sum(cv2.absdiff(f, g)) / f.size
    def check_shot(self, id, time, frame):
        n = id
        # push back diff
        if n == 0:
            self.diff.append(0)
            self.new_shot(id, time, frame)
            return 0
        while len(self.diff) < n:
            self.diff.append(0)
        diff = self.frameFeat[n].diff
        self.diff.append(diff)

        # replace last keyframe/shotboundary with a better one
        lastKeyframe = self.shotList[-1]
        if lastKeyframe.shotId == n-1 and diff > self.diff[n-1]:
            #climbing
            lastKeyframe.shotId = n
            lastKeyframe.id = n
        if lastKeyframe.id == n-1:
            if diff < self.diff[n-1] or self.frameFeat[n].bright1 > self.frameFeat[n-1].bright1:
                self.replace_keyframe(id, time, frame)
        elif lastKeyframe.id > n-6:
            keyFrameDiff = self.frameFeat[lastKeyframe.id].diff
            peakDiff = self.frameFeat[lastKeyframe.shotId].diff
            if keyFrameDiff * keyFrameDiff > peakDiff * diff:
                self.replace_keyframe(id, time, frame)
        
        # pd
        if n > 1 and self.diff[n-1] > self.diff[n-2] and self.diff[n-1] > diff:
            self.diff[n-1] = max(self.diff[n-2], diff) # clipping
        m = n # max radius
        i = n-1
        while i >= n-m:
            dist = math.sqrt((n-i) ** 2 + (diff - self.diff[i]) ** 2) if self.diff[i] < diff else n-i
            if m > dist:
                m = dist
            i -= 1
        if diff > self.diff[n-1] and m > self.frameFeat[n-1].shot:
            self.frameFeat[n-1].shot=1
        if self.frameFeat[n-1].shot > 8 and m < 8:
            self.new_shot(id, time, frame)
        return m
import os
class FrameDumper(FrameCallback):
    def __init__(self, path, ext = '.jpg'):
        super(FrameDumper, self).__init__()
        self.path = path
        self.ext = ext if ext[0] == '.' else '.' + ext
        if not os.path.isdir(path):
            os.mkdir(path)
    def OnFrame(self, id, time, frame):
        fn = os.path.join(self.path, str(id).rjust(5, '0')+ self.ext)
        cv2.imwrite(fn, frame)
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

    dumper = FrameDumper(Args.outpath, Args.fileformat)

    parser.AddCallback(sampler)
    sampler.AddCallback(segmentor)
    segmentor.AddCallback(dumper)

    parser.Run()
