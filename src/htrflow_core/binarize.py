import cv2

from htrflow_core.tasks.base_task import BaseTask


class Binarize(BaseTask):
    def __init__(self):
        pass

    def run(self, img):

        img_ori = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
        dst = cv2.fastNlMeansDenoising(img_gray, h=31, templateWindowSize=7, searchWindowSize=21)
        img_blur = cv2.medianBlur(dst, 3).astype("uint8")
        threshed = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        img_binarized = cv2.cvtColor(threshed, cv2.COLOR_BGR2RGB)

        return img_binarized


if __name__ == "__main__":
    pass
