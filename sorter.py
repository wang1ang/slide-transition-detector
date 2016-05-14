import cv2
import os
import mediaoutput
import imgcomparison
import argparse


class Slide(object):
    """
    Represents a slide
    """
    def __init__(self, time, img):
        """
        Default initializer for a slide representation
        :param time: the time when the slide appears
        :param img: the image representing the slide
        """
        self.time, _ = os.path.splitext(time)
        self.img = img
        self.marked = False
        self.times = []

    def add_time(self, time):
        """
        Add an additional instance in time, when the slide
        is displayed.
        :param time: the time when the slide is displayed
        """
        self.times.append(time)


class SlideDataHelper(object):
    """
    The helps to get slides from data.
    """
    def __init__(self, path):
        """
        Default initializer
        :param path: the path, where the slide is stored on disk
        """
        self.path = path

    def get_slides(self):
        """
        Gets the slide from disk and returns them as list of "Slide"
        objects.
        :return: The slides stored on disk as list of "Slide" objects.
        """
        slides = []
        for filename in sorted(os.listdir(self.path)):
            file_path = os.path.join(self.path, filename)
            _, ext = os.path.splitext(file_path)
            if not is_image(ext):
                continue
            slide = Slide(filename, cv2.imread(file_path))
            slides.append(slide)

        return slides


class SlideSorter(object):
    """
    Sorts the slides according to their timestamp.
    """

    def __init__(self, path, comparator, outpath, timetable_loc, file_format):
        """
        Default initializer
        :param path: the path where the slides are located on disk
        :param comparator: the comparator to determine, if two slides
        are duplicates.
        """
        self.comparator = comparator
        self.inpath = path
        self.outpath = outpath
        self.timetable_loc = timetable_loc
        self.file_format = file_format

    def sort(self):
        """
        Sorting the slides and write the new slides without duplicates
        but with a timetable to disk.
        """
        slides = SlideDataHelper(self.inpath).get_slides()
        unique_slides = self.group_slides(slides)

        mediaoutput.setup_dirs(self.timetable_loc)
        timetable = open(self.timetable_loc, 'w')
        mediaoutput.TimetableWriter(self.outpath, timetable, self.file_format).write(unique_slides)
        timetable.close()

    def group_slides(self, slides):
        """
        Groups the slides by eliminating duplicates.
        :param slides: the list of slides possibly containing duplicates
        :return: a list of slides without duplicates
        """
        for i in xrange(len(slides)):
            slide = slides[i]
            if slide.marked:
                continue

            for j in xrange(i, len(slides)):
                other = slides[j]
                if slide == other or other.marked:
                    continue

                if self.comparator.are_same(slide.img, other.img):
                    slide.add_time(other.time)
                    other.marked = True

        unique_slides = filter(lambda x: not x.marked, slides)

        return unique_slides




def is_image(ext):
    """
    Checks if the file_format is a supported image to read.
    :param ext: the extension of a file.
    :return: whether or not the file is a image
    """
    return ext == '.jpeg' or ext == '.png' or ext == '.jpg' or ext == '.bmp'

if __name__ == '__main__':

    Parser = argparse.ArgumentParser(description="Slide Sorter")
    Parser.add_argument("-d", "--inputslides", help="path of the sequentially sorted slides")
    Parser.add_argument("-o", "--outpath", help="path to output slides", default="unique/", nargs='?')
    Parser.add_argument("-f", "--fileformat", help="file format of the output images e.g. '.jpg'",
                        default=".jpg", nargs='?')
    Parser.add_argument("-t", "--timetable",
                        help="path where the timetable should be written (default is the outpath+'timetable.txt')",
                        nargs='?', default=None)
    Args = Parser.parse_args()
    if Args.timetable is None:
        Args.timetable = os.path.join(Args.outpath, "timetable.txt")

    SlideSorter(Args.inputslides, imgcomparison.AbsDiffHistComparator(0.99), Args.outpath, Args.timetable, Args.fileformat).sort()
    #SlideParser('unique/', 'unique/timetable.txt').parse()

    cv2.destroyAllWindows()
