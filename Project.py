from ImageClass import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.color import *
from PIL import Image


class Ui_ImageProcessingProject(object):
    def setupUi(self, ImageProcessingProject):
        ImageProcessingProject.setObjectName("ImageProcessingProject")
        ImageProcessingProject.resize(826, 600)
        self.centralwidget = QtWidgets.QWidget(ImageProcessingProject)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(30, 70, 741, 141))
        self.tabWidget.setObjectName("tabWidget")
        self.analyseElementaire = QtWidgets.QWidget()
        self.analyseElementaire.setObjectName("analyseElementaire")
        self.negatifButton = QtWidgets.QPushButton(self.analyseElementaire)
        self.negatifButton.setGeometry(QtCore.QRect(10, 10, 75, 23))
        self.negatifButton.setObjectName("negatifButton")
        self.rotationButton = QtWidgets.QPushButton(self.analyseElementaire)
        self.rotationButton.setGeometry(QtCore.QRect(190, 80, 75, 23))
        self.rotationButton.setObjectName("rotationButton")
        self.angleField = QtWidgets.QLineEdit(self.analyseElementaire)
        self.angleField.setGeometry(QtCore.QRect(60, 80, 113, 20))
        self.angleField.setObjectName("angleField")
        self.redemensionnerButton = QtWidgets.QPushButton(self.analyseElementaire)
        self.redemensionnerButton.setGeometry(QtCore.QRect(480, 80, 91, 23))
        self.redemensionnerButton.setObjectName("redemensionnerButton")
        self.pourcentageField = QtWidgets.QLineEdit(self.analyseElementaire)
        self.pourcentageField.setGeometry(QtCore.QRect(350, 80, 113, 20))
        self.pourcentageField.setObjectName("pourcentageField")
        self.label_4 = QtWidgets.QLabel(self.analyseElementaire)
        self.label_4.setGeometry(QtCore.QRect(10, 80, 31, 16))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.analyseElementaire)
        self.label_5.setGeometry(QtCore.QRect(280, 80, 61, 16))
        self.label_5.setObjectName("label_5")
        self.histogrammeButton = QtWidgets.QPushButton(self.analyseElementaire)
        self.histogrammeButton.setGeometry(QtCore.QRect(100, 10, 81, 23))
        self.histogrammeButton.setObjectName("histogrammeButton")
        self.egalisationButton = QtWidgets.QPushButton(self.analyseElementaire)
        self.egalisationButton.setGeometry(QtCore.QRect(200, 10, 81, 23))
        self.egalisationButton.setObjectName("egalisationButton")
        self.etirementButton = QtWidgets.QPushButton(self.analyseElementaire)
        self.etirementButton.setGeometry(QtCore.QRect(300, 10, 81, 23))
        self.etirementButton.setObjectName("etirementButton")
        self.tabWidget.addTab(self.analyseElementaire, "")
        self.binarisation = QtWidgets.QWidget()
        self.binarisation.setObjectName("binarisation")
        self.binarisationOtsuButton = QtWidgets.QPushButton(self.binarisation)
        self.binarisationOtsuButton.setGeometry(QtCore.QRect(270, 10, 81, 23))
        self.binarisationOtsuButton.setObjectName("binarisationOtsuButton")
        self.binarisationManuelleButton = QtWidgets.QPushButton(self.binarisation)
        self.binarisationManuelleButton.setGeometry(QtCore.QRect(160, 10, 75, 23))
        self.binarisationManuelleButton.setObjectName("binarisationManuelleButton")
        self.seuilBinarisation = QtWidgets.QLineEdit(self.binarisation)
        self.seuilBinarisation.setGeometry(QtCore.QRect(40, 10, 113, 20))
        self.seuilBinarisation.setObjectName("seuilBinarisation")
        self.label_6 = QtWidgets.QLabel(self.binarisation)
        self.label_6.setGeometry(QtCore.QRect(10, 10, 21, 16))
        self.label_6.setObjectName("label_6")
        self.tabWidget.addTab(self.binarisation, "")
        self.filtrage = QtWidgets.QWidget()
        self.filtrage.setObjectName("filtrage")
        self.medianButton3 = QtWidgets.QPushButton(self.filtrage)
        self.medianButton3.setGeometry(QtCore.QRect(160, 60, 75, 23))
        self.medianButton3.setObjectName("medianButton3")
        self.gaussienButton8 = QtWidgets.QPushButton(self.filtrage)
        self.gaussienButton8.setGeometry(QtCore.QRect(250, 20, 75, 23))
        self.gaussienButton8.setObjectName("gaussienButton8")
        self.moyenneurButton3 = QtWidgets.QPushButton(self.filtrage)
        self.moyenneurButton3.setGeometry(QtCore.QRect(160, 40, 75, 23))
        self.moyenneurButton3.setObjectName("moyenneurButton3")
        self.moyenneurButton5 = QtWidgets.QPushButton(self.filtrage)
        self.moyenneurButton5.setGeometry(QtCore.QRect(250, 40, 75, 23))
        self.moyenneurButton5.setObjectName("moyenneurButton5")
        self.medianButton5 = QtWidgets.QPushButton(self.filtrage)
        self.medianButton5.setGeometry(QtCore.QRect(250, 60, 75, 23))
        self.medianButton5.setObjectName("medianButton5")
        self.gaussienButton1 = QtWidgets.QPushButton(self.filtrage)
        self.gaussienButton1.setGeometry(QtCore.QRect(160, 20, 75, 23))
        self.gaussienButton1.setObjectName("gaussienButton1")
        self.label = QtWidgets.QLabel(self.filtrage)
        self.label.setGeometry(QtCore.QRect(20, 20, 61, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.filtrage)
        self.label_2.setGeometry(QtCore.QRect(20, 40, 61, 16))
        self.label_2.setObjectName("label_2")
        self.label_8 = QtWidgets.QLabel(self.filtrage)
        self.label_8.setGeometry(QtCore.QRect(20, 60, 51, 16))
        self.label_8.setObjectName("label_8")
        self.tabWidget.addTab(self.filtrage, "")
        self.contour = QtWidgets.QWidget()
        self.contour.setObjectName("contour")
        self.contourGradientButton = QtWidgets.QPushButton(self.contour)
        self.contourGradientButton.setGeometry(QtCore.QRect(10, 10, 75, 23))
        self.contourGradientButton.setObjectName("contourGradientButton")
        self.contourSobelButton = QtWidgets.QPushButton(self.contour)
        self.contourSobelButton.setGeometry(QtCore.QRect(100, 10, 75, 23))
        self.contourSobelButton.setObjectName("contourSobelButton")
        self.contourKirschButton = QtWidgets.QPushButton(self.contour)
        self.contourKirschButton.setGeometry(QtCore.QRect(190, 10, 75, 23))
        self.contourKirschButton.setObjectName("contourKirschButton")
        self.contourRobinsonButton = QtWidgets.QPushButton(self.contour)
        self.contourRobinsonButton.setGeometry(QtCore.QRect(280, 10, 75, 23))
        self.contourRobinsonButton.setObjectName("contourRobinsonButton")
        self.contourLaplacienButton = QtWidgets.QPushButton(self.contour)
        self.contourLaplacienButton.setGeometry(QtCore.QRect(370, 10, 75, 23))
        self.contourLaplacienButton.setObjectName("contourLaplacienButton")
        self.tabWidget.addTab(self.contour, "")
        self.morphologie = QtWidgets.QWidget()
        self.morphologie.setObjectName("morphologie")
        self.morphologieDilatationButton = QtWidgets.QPushButton(self.morphologie)
        self.morphologieDilatationButton.setGeometry(QtCore.QRect(100, 10, 75, 23))
        self.morphologieDilatationButton.setObjectName("morphologieDilatationButton")
        self.morphologieFermetureButton = QtWidgets.QPushButton(self.morphologie)
        self.morphologieFermetureButton.setGeometry(QtCore.QRect(280, 10, 75, 23))
        self.morphologieFermetureButton.setObjectName("morphologieFermetureButton")
        self.morphologieErosionButton = QtWidgets.QPushButton(self.morphologie)
        self.morphologieErosionButton.setGeometry(QtCore.QRect(10, 10, 75, 23))
        self.morphologieErosionButton.setObjectName("morphologieErosionButton")
        self.morphologieOuvertureButton = QtWidgets.QPushButton(self.morphologie)
        self.morphologieOuvertureButton.setGeometry(QtCore.QRect(190, 10, 75, 23))
        self.morphologieOuvertureButton.setObjectName("morphologieOuvertureButton")
        self.tabWidget.addTab(self.morphologie, "")
        self.segmentation = QtWidgets.QWidget()
        self.segmentation.setObjectName("segmentation")
        self.croissanceRegionsButton = QtWidgets.QPushButton(self.segmentation)
        self.croissanceRegionsButton.setGeometry(QtCore.QRect(10, 10, 131, 23))
        self.croissanceRegionsButton.setObjectName("croissanceRegionsButton")
        self.partitionRegionsButton = QtWidgets.QPushButton(self.segmentation)
        self.partitionRegionsButton.setGeometry(QtCore.QRect(160, 10, 111, 23))
        self.partitionRegionsButton.setObjectName("partitionRegionsButton")
        self.kMeansButton = QtWidgets.QPushButton(self.segmentation)
        self.kMeansButton.setGeometry(QtCore.QRect(290, 10, 121, 23))
        self.kMeansButton.setObjectName("kMeansButton")
        self.tabWidget.addTab(self.segmentation, "")
        self.pointInteret = QtWidgets.QWidget()
        self.pointInteret.setObjectName("pointInteret")
        self.houghButton = QtWidgets.QPushButton(self.pointInteret)
        self.houghButton.setGeometry(QtCore.QRect(290, 10, 121, 23))
        self.houghButton.setObjectName("houghButton")
        self.siftButton = QtWidgets.QPushButton(self.pointInteret)
        self.siftButton.setGeometry(QtCore.QRect(10, 10, 131, 23))
        self.siftButton.setObjectName("siftButton")
        self.harrisButton = QtWidgets.QPushButton(self.pointInteret)
        self.harrisButton.setGeometry(QtCore.QRect(160, 10, 111, 23))
        self.harrisButton.setObjectName("harrisButton")
        self.tabWidget.addTab(self.pointInteret, "")
        self.imageInitiale = QtWidgets.QLabel(self.centralwidget)
        self.imageInitiale.setGeometry(QtCore.QRect(80, 220, 301, 271))
        self.imageInitiale.setAutoFillBackground(True)
        self.imageInitiale.setText("")
        self.imageInitiale.setObjectName("imageInitiale")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(393, 220, 20, 271))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.imageTraitee = QtWidgets.QLabel(self.centralwidget)
        self.imageTraitee.setGeometry(QtCore.QRect(420, 220, 301, 271))
        self.imageTraitee.setAutoFillBackground(True)
        self.imageTraitee.setText("")
        self.imageTraitee.setObjectName("imageTraitee")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(30, 20, 341, 21))
        font = QtGui.QFont()
        font.setFamily("Oswald")
        font.setPointSize(12)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(170, 500, 141, 31))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(510, 500, 141, 31))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        ImageProcessingProject.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(ImageProcessingProject)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 826, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        ImageProcessingProject.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(ImageProcessingProject)
        self.statusbar.setObjectName("statusbar")
        ImageProcessingProject.setStatusBar(self.statusbar)
        self.actionNew = QtWidgets.QAction(ImageProcessingProject)
        self.actionNew.setObjectName("actionNew")
        self.actionSave = QtWidgets.QAction(ImageProcessingProject)
        self.actionSave.setObjectName("actionSave")
        self.menuFile.addAction(self.actionNew)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionSave)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(ImageProcessingProject)
        self.tabWidget.setCurrentIndex(6)
        QtCore.QMetaObject.connectSlotsByName(ImageProcessingProject)

    def retranslateUi(self, ImageProcessingProject):
        _translate = QtCore.QCoreApplication.translate
        ImageProcessingProject.setWindowTitle(_translate("ImageProcessingProject", "MainWindow"))
        self.negatifButton.setText(_translate("ImageProcessingProject", "Negatif"))
        self.rotationButton.setText(_translate("ImageProcessingProject", "Rotation"))
        self.redemensionnerButton.setText(_translate("ImageProcessingProject", "Redemensionner"))
        self.label_4.setText(_translate("ImageProcessingProject", "Angle"))
        self.label_5.setText(_translate("ImageProcessingProject", "Pourcentage"))
        self.histogrammeButton.setText(_translate("ImageProcessingProject", "Histogramme"))
        self.egalisationButton.setText(_translate("ImageProcessingProject", "Egalisation"))
        self.etirementButton.setText(_translate("ImageProcessingProject", "Etirement"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.analyseElementaire), _translate("ImageProcessingProject", "Analyse Elementaire"))
        self.binarisationOtsuButton.setText(_translate("ImageProcessingProject", "Otsu"))
        self.binarisationManuelleButton.setText(_translate("ImageProcessingProject", "Manuelle"))
        self.label_6.setText(_translate("ImageProcessingProject", "Seuil"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.binarisation), _translate("ImageProcessingProject", "Binarisation"))
        self.medianButton3.setText(_translate("ImageProcessingProject", "3*3"))
        self.gaussienButton8.setText(_translate("ImageProcessingProject", "0.8"))
        self.moyenneurButton3.setText(_translate("ImageProcessingProject", "3*3"))
        self.moyenneurButton5.setText(_translate("ImageProcessingProject", "5*5"))
        self.medianButton5.setText(_translate("ImageProcessingProject", "5*5"))
        self.gaussienButton1.setText(_translate("ImageProcessingProject", "0.1"))
        self.label.setText(_translate("ImageProcessingProject", "Gaussien"))
        self.label_2.setText(_translate("ImageProcessingProject", "Moyenneur"))
        self.label_8.setText(_translate("ImageProcessingProject", "Médian"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.filtrage), _translate("ImageProcessingProject", "Filtrage"))
        self.contourGradientButton.setText(_translate("ImageProcessingProject", "Gradient"))
        self.contourSobelButton.setText(_translate("ImageProcessingProject", "Sobel"))
        self.contourKirschButton.setText(_translate("ImageProcessingProject", "Kirsch"))
        self.contourRobinsonButton.setText(_translate("ImageProcessingProject", "Robinson"))
        self.contourLaplacienButton.setText(_translate("ImageProcessingProject", "Laplacien"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.contour), _translate("ImageProcessingProject", "Contour"))
        self.morphologieDilatationButton.setText(_translate("ImageProcessingProject", "Dilatation"))
        self.morphologieFermetureButton.setText(_translate("ImageProcessingProject", "Fermeture"))
        self.morphologieErosionButton.setText(_translate("ImageProcessingProject", "Erosion"))
        self.morphologieOuvertureButton.setText(_translate("ImageProcessingProject", "Ouverture"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.morphologie), _translate("ImageProcessingProject", "Morphologie"))
        self.croissanceRegionsButton.setText(_translate("ImageProcessingProject", "Croissance de régions"))
        self.partitionRegionsButton.setText(_translate("ImageProcessingProject", "Partition de régions"))
        self.kMeansButton.setText(_translate("ImageProcessingProject", "Méthode des k-means"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.segmentation), _translate("ImageProcessingProject", "Segmentation"))
        self.houghButton.setText(_translate("ImageProcessingProject", "Méthode Hough "))
        self.siftButton.setText(_translate("ImageProcessingProject", "Méthode Sift"))
        self.harrisButton.setText(_translate("ImageProcessingProject", "Méthode Harris"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.pointInteret), _translate("ImageProcessingProject", "Point d\'interet"))
        self.label_3.setText(_translate("ImageProcessingProject", "Image-Proccessing  Project | Firass Mohammed"))
        self.label_7.setText(_translate("ImageProcessingProject", "image initiale"))
        self.label_9.setText(_translate("ImageProcessingProject", "image traitée"))
        self.menuFile.setTitle(_translate("ImageProcessingProject", "File"))
        self.actionNew.setText(_translate("ImageProcessingProject", "New"))
        self.actionSave.setText(_translate("ImageProcessingProject", "Save"))

       # connect buttons to the functions below :

        self.actionNew.triggered.connect(self.openFile)
        self.actionSave.triggered.connect(self.enregistrementimage)

        self.binarisationOtsuButton.clicked.connect(self.BinarisationOtsu)
        self.binarisationManuelleButton.clicked.connect(self.BinarisationLocal)

        self.negatifButton.clicked.connect(self.negatif)
        self.egalisationButton.clicked.connect(self.egalisation)

        self.etirementButton.clicked.connect(self.etir)

        self.histogrammeButton.clicked.connect(self.histo)

        self.rotationButton.clicked.connect(self.rotate)
        self.redemensionnerButton.clicked.connect(self.redim)

        # filtrage

        self.gaussienButton1.clicked.connect(self.gaussian1)
        self.gaussienButton8.clicked.connect(self.gaussian8)

        self.moyenneurButton3.clicked.connect(self.Moyenneur3)
        self.moyenneurButton5.clicked.connect(self.Moyenneur5)

        self.medianButton3.clicked.connect(self.median3)
        self.medianButton5.clicked.connect(self.median5)

        # contour

        self.contourGradientButton.clicked.connect(self.grad)
        self.contourSobelButton.clicked.connect(self.Sobel)
        self.contourRobinsonButton.clicked.connect(self.Robinson)
        self.contourKirschButton.clicked.connect(self.Kirsch)

        self.contourLaplacienButton.clicked.connect(self.laplacien)

        # Morphologie

        self.morphologieErosionButton.clicked.connect(self.Erosion)
        self.morphologieDilatationButton.clicked.connect(self.dilatation)

        self.morphologieOuvertureButton.clicked.connect(self.ouverture)
        self.morphologieFermetureButton.clicked.connect(self.fermeture)

        # segmentation
        self.kMeansButton.clicked.connect(self.kmeans)
        self.partitionRegionsButton.clicked.connect(self.partRegion)
        self.croissanceRegionsButton.clicked.connect(self.croissanceRegion)

        # les points d'interets

        self.siftButton.clicked.connect(self.sift)
        self.harrisButton.clicked.connect(self.harris)
        self.houghButton.clicked.connect(self.Hough)

    # openFile method :

    def openFile(self):

        nom_fichier = QFileDialog.getOpenFileName()
        self.path = nom_fichier[0]
        pathx = self.path
        pixmap = QtGui.QPixmap(pathx)
        pixmap4 = pixmap.scaled(351, 341, QtCore.Qt.KeepAspectRatio)

        self.imageInitiale.setPixmap(QtGui.QPixmap(pixmap4))

    def enregistrementimage(self):
        fileName = QFileDialog.getSaveFileName(
            None, 'some text', "untitled.png", "Image Files (*.jpg *.gif *.bmp *.png)")
        self.fileName = fileName[0]
        print(fileName[0]+'ssss')
        cv2.imwrite(fileName[0], self.mat)

    def rotate(self):
        anglevalue = int(self.angleField.text())
        image = cv2.imread(self.path)
        o = ImageClass(image)
        img = o.rotate_image(anglevalue)
        self.mat = img
        height, width, byteValue = img.shape
        print(byteValue)
        if byteValue == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imag = QtGui.QImage(img, width, height, byteValue *
                                width, QtGui.QImage.Format_RGB888)
        else:
            imag = QtGui.QImage(
                img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(351, 341, QtCore.Qt.KeepAspectRatio)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def redim(self):
        image = cv2.imread(self.path)
        pourcentage = int(self.pourcentageField.text())
        scale_percent = pourcentage
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        self.mat = img
        height, width, byteValue = img.shape
        print(byteValue)
        if byteValue == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imag = QtGui.QImage(img, width, height, byteValue *
                                width, QtGui.QImage.Format_RGB888)
        else:
            imag = QtGui.QImage(
                img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(351, 341)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def negatif(self):
        image = cv2.imread(self.path)
        img = 255 - image
        self.mat = img
        height, width, byteValue = img.shape
        print(byteValue)
        if byteValue == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imag = QtGui.QImage(img, width, height, byteValue *
                                width, QtGui.QImage.Format_RGB888)
        else:
            imag = QtGui.QImage(
                img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(351, 341, QtCore.Qt.KeepAspectRatio)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def Moyenneur5(self):
        image = cv2.imread(self.path)
        f = ImageClass(image)
        img = f.Moyenneur(5)
        self.mat = img
        height, width, byteValue = img.shape
        print(byteValue)
        if byteValue == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imag = QtGui.QImage(img, width, height, byteValue *
                                width, QtGui.QImage.Format_RGB888)
        else:
            imag = QtGui.QImage(
                img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(351, 341, QtCore.Qt.KeepAspectRatio)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def Moyenneur3(self):
        image = cv2.imread(self.path)
        f = ImageClass(image)
        img = f.Moyenneur(3)
        self.mat = img
        height, width, byteValue = img.shape
        print(byteValue)
        if byteValue == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imag = QtGui.QImage(img, width, height, byteValue *
                                width, QtGui.QImage.Format_RGB888)
        else:
            imag = QtGui.QImage(
                img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(351, 341, QtCore.Qt.KeepAspectRatio)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def gaussian1(self):
        image = cv2.imread(self.path)
        f = ImageClass(image)
        img = f.Gaussien(0.1)
        self.mat = img
        height, width, byteValue = img.shape
        print(byteValue)
        if byteValue == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imag = QtGui.QImage(img, width, height, byteValue *
                                width, QtGui.QImage.Format_RGB888)
        else:
            imag = QtGui.QImage(
                img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(351, 341, QtCore.Qt.KeepAspectRatio)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def gaussian8(self):
        image = cv2.imread(self.path)
        f = ImageClass(image)
        img = f.Gaussien(0.8)
        self.mat = img
        height, width, byteValue = img.shape
        print(byteValue)
        if byteValue == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imag = QtGui.QImage(img, width, height, byteValue *
                                width, QtGui.QImage.Format_RGB888)
        else:
            imag = QtGui.QImage(
                img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(351, 341, QtCore.Qt.KeepAspectRatio)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def median3(self):
        image = cv2.imread(self.path)
        height, width, byteValue = image.shape
        if byteValue == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            f = ImageClass(image)
            img = f.Median(3)
            self.mat = img
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            imag = QtGui.QImage(img, width, height, byteValue *
                                width, QtGui.QImage.Format_RGB888)
        else:
            f = ImageClass(image)
            img = f.Median(3)
            self.mat = img
            imag = QtGui.QImage(
                img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(351, 341, QtCore.Qt.KeepAspectRatio)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def median5(self):
        image = cv2.imread(self.path)
        height, width, byteValue = image.shape
        if byteValue == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            f = ImageClass(image)
            img = f.Median(5)
            self.mat = img
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            imag = QtGui.QImage(img, width, height, byteValue *
                                width, QtGui.QImage.Format_RGB888)
        else:
            f = ImageClass(image)
            img = f.Median(5)
            self.mat = img
            imag = QtGui.QImage(
                img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(351, 341, QtCore.Qt.KeepAspectRatio)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def grad(self):
        image = cv2.imread(self.path)
        height, width, byteValue = image.shape
        if byteValue == 3:
            imag = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            c = ImageClass(imag)
            img = c.grad(20)
            self.mat = img
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            imag = QtGui.QImage(img, width, height, byteValue *
                                width, QtGui.QImage.Format_RGB888)
        else:
            c = ImageClass(image)
            img = c.grad(20)
            self.mat = img
            imag = QtGui.QImage(
                img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)

        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(351, 341, QtCore.Qt.KeepAspectRatio)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def Robert(self):
        image = cv2.imread(self.path)
        height, width, byteValue = image.shape
        print(byteValue)
        if byteValue == 3:
            imag = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            c = ImageClass(imag)
            img = c.Robert(20)
            self.mat = img
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            imag = QtGui.QImage(img, width, height, byteValue *
                                width, QtGui.QImage.Format_RGB888)
        else:
            c = ImageClass(image)
            img = c.Robert(20)
            self.mat = img
            imag = QtGui.QImage(
                img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)

        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(351, 341, QtCore.Qt.KeepAspectRatio)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def Sobel(self):
        image = cv2.imread(self.path)
        height, width, byteValue = image.shape
        print(byteValue)
        if byteValue == 3:
            imag = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            c = ImageClass(imag)
            img = c.Sobel(50)
            self.mat = img
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            imag = QtGui.QImage(img, width, height, byteValue *
                                width, QtGui.QImage.Format_RGB888)
        else:
            c = ImageClass(image)
            img = c.Sobel(50)
            self.mat = img
            imag = QtGui.QImage(
                img, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)

        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(351, 341, QtCore.Qt.KeepAspectRatio)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def laplacien(self):
        image = cv2.imread(self.path)
        height, width, byteValue = image.shape
        if byteValue == 3:
            imag = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            c = ImageClass(imag)
            img = c.Laplacien(20)
            self.mat = img
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            imag = QtGui.QImage(img, width, height, byteValue *
                                width, QtGui.QImage.Format_RGB888)
        else:
            c = ImageClass(image)
            img = c.Laplacien(20)
            self.mat = img
            imag = QtGui.QImage(
                img, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(351, 341, QtCore.Qt.KeepAspectRatio)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def Erosion(self):
        image = cv2.imread(self.path)
        height, width, byteValue = image.shape
        print(byteValue)
        if byteValue == 3:
            imag = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            m = ImageClass(imag)
            h = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
            img = m.Erosion(h)
            self.mat = img
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            imag = QtGui.QImage(img, width, height, byteValue *
                                width, QtGui.QImage.Format_RGB888)
        else:
            m = ImageClass(image)
            h = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
            img = m.Erosion(h)
            self.mat = img
            imag = QtGui.QImage(
                img, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(351, 341)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def dilatation(self):
        image = cv2.imread(self.path)
        height, width, byteValue = image.shape
        print(byteValue)
        if byteValue == 3:
            imag = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            m = ImageClass(imag)
            h = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
            img = m.dilatation(h)
            self.mat = img
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            imag = QtGui.QImage(img, width, height, byteValue *
                                width, QtGui.QImage.Format_RGB888)
        else:
            m = ImageClass(image)
            h = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
            img = m.dilatation(h)
            self.mat = img
            imag = QtGui.QImage(
                img, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(351, 341, QtCore.Qt.KeepAspectRatio)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def ouverture(self):
        image = cv2.imread(self.path)
        height, width, byteValue = image.shape
        print(byteValue)
        if byteValue == 3:
            imag = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            m = ImageClass(imag)
            h = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
            imaag = m.Erosion(h)
            m1 = ImageClass(imaag)
            h = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
            img = m1.dilatation(h)
            self.mat = img
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            imag = QtGui.QImage(img, width, height, byteValue *
                                width, QtGui.QImage.Format_RGB888)
        else:
            m = ImageClass(image)
            h = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
            imaag = m.Erosion(h)
            m1 = ImageClass(imaag)
            h = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
            img = m1.dilatation(h)
            self.mat = img
            imag = QtGui.QImage(
                img, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(351, 341, QtCore.Qt.KeepAspectRatio)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def fermeture(self):
        image = cv2.imread(self.path)
        height, width, byteValue = image.shape
        print(byteValue)
        if byteValue == 3:
            imag = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            m = ImageClass(imag)
            h = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
            imaag = m.dilatation(h)
            m1 = ImageClass(imaag)
            h = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
            img = m1.Erosion(h)
            self.mat = img
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            imag = QtGui.QImage(img, width, height, byteValue *
                                width, QtGui.QImage.Format_RGB888)
        else:
            m = ImageClass(image)
            h = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
            imaag = m.dilatation(h)
            m1 = ImageClass(imaag)
            h = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
            img = m1.Erosion(h)
            self.mat = img
            imag = QtGui.QImage(
                img, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(351, 341, QtCore.Qt.KeepAspectRatio)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def BinarisationOtsu(self):
        image = cv2.imread(self.path)
        height, width, byteValue = image.shape
        if byteValue == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            b = ImageClass(image)
            print('hello')
            img = b.Otsu()
            self.mat = img
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            imag = QtGui.QImage(img, width, height, byteValue *
                                width, QtGui.QImage.Format_RGB888)
        else:
            b = ImageClass(image)
            img = b.Otsu()
            self.mat = img
            imag = QtGui.QImage(
                img, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(351, 341, QtCore.Qt.KeepAspectRatio)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def BinarisationLocal(self):
        name = int(self.seuilBinarisation.text())
        image = cv2.imread(self.path)
        height, width, byteValue = image.shape
        print(byteValue)
        if byteValue == 3:
            imag = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            f = ImageClass(imag)
            img = f.Seuillage(name)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            imag = QtGui.QImage(
                img, width, height, byteValue * width, QtGui.QImage.Format_RGB888)
        else:
            f = ImageClass(image)
            img = f.Seuillage(name)
            imag = QtGui.QImage(
                img, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(381, 341)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def kmeans(self):
        image = cv2.imread(self.path)
        imag = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, byteValue = imag.shape
        s = ImageClass(imag)
        img = s.k_means()
        self.mat = img
        imag = QtGui.QImage(img, width, height, byteValue *
                            width, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(351, 341, QtCore.Qt.KeepAspectRatio)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def partRegion(self):
        image = cv2.imread(self.path)
        imag = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, byteValue = imag.shape
        s = ImageClass(imag)
        img = s.partition_regions()
        imag = QtGui.QImage(img, width, height, byteValue *
                            width, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(351, 341, QtCore.Qt.KeepAspectRatio)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def histo(self):
        image = cv2.imread(self.path)
        o = ImageClass(image)
        o.hist()

    def egalisation(self):
        image = cv2.imread(self.path)
        imag = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        o = ImageClass(imag)
        img = o.histeq()
        self.mat = img
        imag = QtGui.QImage(
            img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(351, 341, QtCore.Qt.KeepAspectRatio)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def etir(self):
        image = cv2.imread(self.path)
        imag = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        o = ImageClass(imag)
        img = o.etire()
        self.mat = img
        imag = QtGui.QImage(
            img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(351, 341, QtCore.Qt.KeepAspectRatio)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def sift(self):
        print('sift')

    def harris(self):
        print('harris')

    def Hough(self):
        print('hough')

    def Robinson(self):
        print('Robinson')

    def Kirsch(self):
        print('Kirsch')

    def croissanceRegion(self):
        print('croissanceRegion')


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_ImageProcessingProject()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
