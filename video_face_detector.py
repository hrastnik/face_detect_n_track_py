import cv2

class VideoFaceDetector(object):

    def __init__(self, cascadeFilePath, videoCapture):
        self._videoCapture = None
        self._faceCascade = None
        self._allFaces = []
        self._trackedFace = None
        self._faceRoi = None
        self._faceTemplate = None
        self._matchingResult = None
        self._templateMatchingRunning = False
        self._templateMatchingStartTime = 0
        self._templateMatchingCurrentTime = 0
        self._foundFace = False
        self._scale = 0
        self._resizedWidth = 320
        self._facePosition = None
        self._templateMatchingMaxDuration = 3
        
        self.setFaceCascade(cascadeFilePath)
        self.setVideoCapture(videoCapture)    


    def setVideoCapture(self, videoCapture):
        self._videoCapture = videoCapture


    def videoCapture(self, ):
        return self._videoCapture


    def setFaceCascade(self, cascadeFilePath):
        if self._faceCascade is None:
            self._faceCascade = cv2.CascadeClassifier(cascadeFilePath)
        else:
            self._faceCascade.load(cascadeFilePath)


    def faceCascade(self, frame):
        return self._faceCascade


    def setResizedWidth(self, width):
        self._resizedWidth = max(1, width)


    def resizedWidth(self):
        return self.resizedWidth


    def isFaceFound(self):
        return self._foundFace


    def face(self):
        x, y, w, h = self._trackedFace
        x /= self._scale
        y /= self._scale
        w /= self._scale
        h /= self._scale
        return (x, y, w, h)

    def facePosition(self):
        x, y = self._facePosition
        x /= self._scale
        y /= self._scale
        return x, y
        
    def setTemplateMatchingMaxDuration(self, s):
        self._templateMatchingMaxDuration = s


    def templateMatchingMaxDuration(self):
        return self._templateMatchingMaxDuration


    def __doubleRectSize(self, inputRect, frameSize):
        xi, yi, wi, hi = inputRect

        wo = wi * 2
        ho = hi * 2
        xo = xi - wi/2
        yo = yi - hi/2

        xf, yf, wf, hf = frameSize

        if xo < xf:
            wo += xo
            xo = xf
        if yo < yf:
            ho += yo
            yo = yf

        if xo + wo > wf:
            wo = wf - xo
        if yo + ho > hf:
            ho = hf - yo
        
        return xo, yo, wo, ho


    def __biggestFace(self, faces):
        return max(faces, key=lambda f: f[2] * f[3])
        

    def __centerOfRect(self, rect):
        x, y, w, h = rect
        return x + w/2, y + h/2


    def __getFaceTemplate(self, frame, face):
        x, y, w, h = face
        x = int(x + w/4)
        y = int(y + h/4)
        w = int(w / 2)
        h = int(h / 2)

        template = frame[y:y+h, x:x+w].copy()
        return template
        

    def __detectFaceAllSizes(self, frame):
        rows, cols = frame.shape[:2]
        self._allFaces = self._faceCascade.detectMultiScale(frame, 1.1, 3, 0,
            (int(rows / 5), int(rows / 5)), (int(rows * 2 / 3), int(rows * 2 / 3)))

        if len(self._allFaces) == 0: 
            self._foundFace = False
            return

        self._foundFace = True
        self._trackedFace = self.__biggestFace(self._allFaces)
        self._faceTemplate = self.__getFaceTemplate(frame, self._trackedFace)
        self._faceRoi = self.__doubleRectSize(self._trackedFace, (0, 0, cols, rows))
        self._facePosition = self.__centerOfRect(self._trackedFace)


    def __detectFaceAroundRoi(self, frame):
        x, y, w, h = [int(i) for i in self._faceRoi]
        roiPatch = frame[y:y+h, x:x+w]
        self._allFaces = self._faceCascade.detectMultiScale(roiPatch, 1.1, 3, 0,
            (int(self._trackedFace[2] * 8 / 10), int(self._trackedFace[3] * 8 / 10)),
            (int(self._trackedFace[2] * 12 / 10), int(self._trackedFace[2] * 12 / 10)))

        if len(self._allFaces) == 0:
            self._templateMatchingRunning = True
            if self._templateMatchingStartTime == 0:
                self._templateMatchingStartTime = cv2.getTickCount()
            return

        self._templateMatchingRunning = False
        self._templateMatchingCurrentTime = 0
        self._templateMatchingStartTime = 0

        xf, yf, wf, hf = self.__biggestFace(self._allFaces)
        self._trackedFace = (xf + x, yf + y, wf, hf)

        self._faceTemplate = self.__getFaceTemplate(frame, self._trackedFace)
        h, w = frame.shape[:2]
        self._faceRoi = self.__doubleRectSize(self._trackedFace, (0, 0, w, h))

        self._facePosition = self.__centerOfRect(self._trackedFace)
            

    def __detectFacesTemplateMatching(self, frame):
        self._templateMatchingCurrentTime = cv2.getTickCount()
        duration = (self._templateMatchingCurrentTime - self._templateMatchingStartTime) / cv2.getTickFrequency()

        rows, cols = self._faceTemplate.shape[:2]

        if duration > self._templateMatchingMaxDuration or rows * cols == 0 or rows <= 1 or cols <= 1:
            self._foundFace = False
            self._templateMatchingRunning = False
            self._templateMatchingStartTime = 0
            self._templateMatchingCurrentTime = 0
            self._facePosition = (0, 0)
            self._trackedFace = (0, 0, 0, 0)
            return


        x, y, w, h = [int(i) for i in self._faceRoi] 
        roiPatch = frame[y:y+h, x:x+w]
        self._matchingResult = cv2.matchTemplate(roiPatch, self._faceTemplate, cv2.TM_SQDIFF_NORMED)
        self._matchingResult = cv2.normalize(self._matchingResult, 0, 1, cv2.NORM_MINMAX)
        
        _, _, minLoc, _ = cv2.minMaxLoc(self._matchingResult)
        x, y = minLoc
        xr, yr = self._faceRoi[:2]
        rows, cols = self._faceTemplate.shape[:2]
        trackedFace = (x + xr, y + yr, cols, rows)

        maxCols, maxRows = frame.shape[:2]
        self._trackedFace = self.__doubleRectSize(trackedFace, (0, 0, maxRows, maxCols))
        self._faceTemplate = self.__getFaceTemplate(frame, self._trackedFace)
        self._faceRoi = self.__doubleRectSize(self._trackedFace, (0, 0, maxCols, maxRows))
        self._facePosition = self.__centerOfRect(self._trackedFace)


    def getFrameAndDetect(self):
        _, frame = self._videoCapture.read()

        rows, cols = frame.shape[:2]

        self._scale = min(self._resizedWidth, cols) / float(cols)
        resizedFrameSize = int(self._scale * cols), int(self._scale * rows)

        resizedFrame = cv2.resize(frame, resizedFrameSize)

        if not self._foundFace:
            self.__detectFaceAllSizes(resizedFrame)
        else:
            self.__detectFaceAroundRoi(resizedFrame)
            if self._templateMatchingRunning:
                self.__detectFacesTemplateMatching(resizedFrame)

        return frame