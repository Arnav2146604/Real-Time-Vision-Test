import sys
from PyQt5 import QtCore
import numpy as np
import cv2
import dlib
import random

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap, QTransform
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread

from PyQt5.uic import loadUi


import requests
import json
from geopy.geocoders import Nominatim
import geocoder


class HomeScreen(QMainWindow):
    def __init__(self, widgets):
        super(HomeScreen, self).__init__()
        loadUi('home.ui', self)

        self.widgets = widgets

        self.btnStart.clicked.connect(self.startVisionAcuity)



    def startVisionAcuity(self):
        self.widgets.setCurrentIndex(widgets.currentIndex() + 1)



class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    change_dist_signal = pyqtSignal(int)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    left = [36, 37, 38, 39, 40, 41]
    right = [42, 43, 44, 45, 46, 47]

    kernel = np.ones((9, 9), np.uint8)

    focal_length = 5076.92
    iris_diameter_cm = 1.17

    is_camera_running = True

    def shape_to_np(self, shape, dtype='int'):
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((68, 2), dtype=dtype)
        # loop over the 68 facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        # return the list of (x, y)-coordinates
        return coords

    def eye_on_mask(self, shape, mask, side):
        points = [shape[i] for i in side]
        points = np.array(points, dtype=np.int32)
        mask = cv2.fillConvexPoly(mask, points, 255)
        return mask

    def stop(self):
        self.is_camera_running = True

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 260)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)
        while self.is_camera_running:
            ret, img = cap.read()
            dist, ctr = 0, 0
            if ret:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rects = self.detector(gray, 1)
                for rect in rects:
                    shape = self.predictor(gray, rect)
                    shape = self.shape_to_np(shape)
                    mask = np.zeros(img.shape[:2], dtype=np.uint8)
                    mask = self.eye_on_mask(shape, mask, self.left)
                    mask = self.eye_on_mask(shape, mask, self.right)
                    mask = cv2.dilate(mask, self.kernel, 5)
                    eyes = cv2.bitwise_and(img, img, mask=mask)
                    mask = (eyes == [0, 0, 0]).all(axis=2)
                    eyes[mask] = [255, 255, 255]
                    iris_diam_px = abs(shape[42][0] - shape[39][0])
                    iris_depth = (self.focal_length *
                                  self.iris_diameter_cm) / iris_diam_px
                    dist += iris_depth
                    ctr += 1

                if ctr == 0:
                    ctr = 1
                dist = dist / ctr
                self.change_pixmap_signal.emit(img)
                self.change_dist_signal.emit(int(dist))




class VisionTestThread(QThread):
    e_sizes = [1.4544, 1.16352, 0.916272, 0.7272,
               0.5816, 0.4654, 0.3636, 0.2909, 0.2327, 0.1818, 0.14544]
    optotypes_read = [0] * 11
    log_mar = 1.0
    ctr = 0
    direction = random.randint(0, 3)
    test_over_signal = pyqtSignal(list)

    def __init__(self, label, dpX, dpY, pixmap, buttons):
        super().__init__()
        self.letter_label = label
        self.dpX = dpX
        self.dpY = dpY
        self.pixmap = pixmap
        self.buttons = buttons

        self.buttons[3].clicked.connect(self.clickedUp)
        self.buttons[0].clicked.connect(self.clickedRight)
        self.buttons[1].clicked.connect(self.clickedDown)
        self.buttons[2].clicked.connect(self.clickedLeft)

    def run(self):
        for i in range(len(self.buttons)):
            self.buttons[i].setVisible(True)
        self.display_letter()

    def display_letter(self):
        if self.ctr >= 55:
            # Finished
            self.test_over_signal.emit(self.optotypes_read)
            return
        new_dir = random.randint(0, 3)
        while new_dir == self.direction:
            new_dir = random.randint(0, 3)
        self.direction = new_dir
        sizeX, sizeY = round(
            self.dpX * self.e_sizes[self.ctr // 5]), round(self.dpY * self.e_sizes[self.ctr // 5])
        cur_pixmap = self.pixmap.scaled(
            sizeX, sizeY, QtCore.Qt.KeepAspectRatio)
        cur_pixmap = cur_pixmap.transformed(
            QTransform().rotate(self.direction * 90), QtCore.Qt.SmoothTransformation)
        self.letter_label.setPixmap(cur_pixmap)

    def read_correct(self):
        if self.ctr >= 55:
            # Finished
            return
        self.optotypes_read[self.ctr // 5] += 1

    def clickedUp(self):
        if self.direction == 3:
            # Correct
            self.read_correct()
        self.ctr += 1
        self.display_letter()

    def clickedRight(self):
        if self.direction == 0:
            # Correct
            self.read_correct()
        self.ctr += 1
        self.display_letter()

    def clickedDown(self):
        if self.direction == 1:
            # Correct
            self.read_correct()
        self.ctr += 1
        self.display_letter()

    def clickedLeft(self):
        if self.direction == 2:
            # Correct
            self.read_correct()
        self.ctr += 1
        self.display_letter()


class VisionAcuity(QMainWindow):
    userSittingImproperly = 20

    def __init__(self, dpX, dpY, widgets):
        super(VisionAcuity, self).__init__()
        loadUi('acuity_dist.ui', self)

        self.widgets = widgets

        self.disply_width = 260
        self.display_height = 220

        self.dpcX, self.dpcY = dpX / 2.54, dpY / 2.54

        # create the video capture thread
        self.video_thread = VideoThread()
        # connect its signal to the update_image slot
        self.video_thread.change_pixmap_signal.connect(self.updateImage)
        self.video_thread.change_dist_signal.connect(self.updateDistLabel)
        # start the thread
        self.video_thread.start()

        # Set the E for reading
        self.echart_label.setVisible(False)
        self.pixmap = QPixmap('./assets/img/tumbling_e/1.png')
        self.visionButtons = [self.btnRight,
                              self.btnDown, self.btnLeft, self.btnUp]
        for i in range(len(self.visionButtons)):
            self.visionButtons[i].setVisible(False)
        self.vision_test_thread = VisionTestThread(
            self.echart_label, self.dpcX, self.dpcY, self.pixmap, self.visionButtons)

        self.btnStart.clicked.connect(self.drawTumblingE)

        self.vision_test_thread.test_over_signal.connect(self.testOver)

    def drawTumblingE(self):
        self.instruction_label1.setVisible(False)
        self.instruction_label2.setVisible(False)
        self.btnStart.setVisible(False)
        self.echart_label.setVisible(True)
        self.vision_test_thread.start()

    @pyqtSlot(int)
    def updateDistLabel(self, dist):
        self.dist_label.setText(f'{dist} cm')
        if dist >= 90 and dist <= 110:
            self.image_label.setStyleSheet('border: 3px solid lightgreen')
            self.userSittingImproperly = 0
        else:
            if self.userSittingImproperly > 5:
                self.image_label.setStyleSheet('border: 3px solid red')
            self.userSittingImproperly += 1

    @pyqtSlot(np.ndarray)
    def updateImage(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(
            rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(
            self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    @pyqtSlot(list)
    def testOver(self, optotypes_read):
        self.vision_test_thread.deleteLater()

        resultsScreen = ResultsScreen(self.widgets, optotypes_read)
        self.widgets.addWidget(resultsScreen)
        self.widgets.setCurrentIndex(self.widgets.currentIndex() + 1)


class ResultsScreen(QMainWindow):
    MAR_values = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]

    def __init__(self, widgets, optotypes_read):
        super(ResultsScreen, self).__init__()
        loadUi('results.ui', self)

        self.widgets = widgets
        self.optotypes_read = optotypes_read
        self.progressBar.setValue(optotypes_read[0])
        self.progressBar_2.setValue(optotypes_read[1])
        self.progressBar_3.setValue(optotypes_read[2])
        self.progressBar_4.setValue(optotypes_read[3])
        self.progressBar_5.setValue(optotypes_read[4])
        self.progressBar_6.setValue(optotypes_read[5])
        self.progressBar_7.setValue(optotypes_read[6])
        self.progressBar_8.setValue(optotypes_read[7])
        self.progressBar_9.setValue(optotypes_read[8])
        self.progressBar_10.setValue(optotypes_read[9])
        self.progressBar_11.setValue(optotypes_read[10])

        MAR_score = self.calculateMAR()
        self.mar_label.setText(str(MAR_score))

        #API call
        g = geocoder.ip('me')

        geolocator = Nominatim(user_agent="geoapiExercises")
        Latitude = f"{g.latlng[0]}"
        Longitude = f"{g.latlng[1]}"

        location = geolocator.reverse(Latitude + "," + Longitude)

        # Display
        # print(location)
        address = location.raw['address']
        # print(address["city"])
        # Set the base URL and query parameters
        base_url = "https://nominatim.openstreetmap.org/search"

        city = address["city"]
        type = f"eye+clinic+in+{city}+india"

        # Make the API request
        url = f"{base_url}?q={type}&format=json&limit=10&accept-language=en"
        response = requests.get(url)

        # Parse the response
        data = json.loads(response.text)

        # Print the names of the doctors that were found
        s="\n"
        for doctor in data:
            s+=doctor["display_name"] + "\n"

        if MAR_score >= 0.4:
            self.status.setText(
                f'Your eyesight is poor. You need to immediately consult a doctor.\n Nearby Doctors:{s}')
            self.status.setStyleSheet('color: red;')
        elif MAR_score >= 0.1:
            self.status.setText(
                f'Your eyesight is weak. Do consult a doctor as soon as possible.\n Nearby Doctors:{s}')
            self.status.setStyleSheet('color: orange;')
        else:
            self.status.setText(
                'You have a perfect MAR score. Your eyes are healthy.')
            self.status.setStyleSheet('color: green;')


    def calculateMAR(self):
        bestMAR, opt_missed = 0.0, 0
        for i in range(len(self.optotypes_read)):
            if self.optotypes_read[i] < 3:
                bestMAR = self.MAR_values[i]
                opt_missed = 5 - self.optotypes_read[i]
                break

        return round(bestMAR + 0.02 * opt_missed, 2)

    def gotoMenu(self):
        self.widgets.setCurrentIndex(1)

    def startVisionAcuity(self):
        self.widgets.setCurrentIndex(widgets.currentIndex() + 1)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    screen = app.screens()[0]
    dpX, dpY = screen.physicalDotsPerInchX(), screen.physicalDotsPerInchY()

    widgets = QtWidgets.QStackedWidget()

    homeScreen = HomeScreen(widgets)
    # menuScreen = MenuScreen(widgets)
    visionAcuityDist = VisionAcuity(dpX, dpY, widgets)

    widgets.addWidget(homeScreen)
    # widgets.addWidget(menuScreen)
    widgets.addWidget(visionAcuityDist)

    widgets.setFixedSize(800, 600)
    widgets.setWindowTitle('Real Time Eye Checkup')

    widgets.show()

    try:
        sys.exit(app.exec_())
    except:
        print('Exiting')
