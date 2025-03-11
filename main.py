import sys
import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout

class HandDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Building the model
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75,
            max_num_hands=2)

        # Initializing the GUI
        self.initUI()

    def initUI(self):
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        # QLabel to display the video feed
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)

        # QLabel to display the hands photo
        self.hands_photo_label = QLabel(self)
        hands_pixmap = QPixmap(r'C:\Users\Alex\PycharmProjects\Right_Left\hands.png')  # Replace with the actual path to your image
        self.hands_photo_label.setPixmap(hands_pixmap)
        self.hands_photo_label.setAlignment(Qt.AlignCenter)

        # QPushButton pentru a porni si opri detectia
        self.start_button = QPushButton('Start Detection', self)
        self.start_button.clicked.connect(self.start_detection)

        # QVBoxLayout to arrange widgets vertically
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.addWidget(self.video_label, 1)  # Add stretch factor to center QLabel
        self.layout.addWidget(self.hands_photo_label, alignment=Qt.AlignCenter)  # Center the hands photo QLabel
        self.layout.addStretch(1)  # Add stretch to push the QPushButton to the bottom
        self.layout.addWidget(self.start_button, alignment=Qt.AlignCenter)  # Center the QPushButton

        # Apply a modern stylesheet to the main window
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ffffff;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                font-size: 16px;
                min-width: 150px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        # Initializing camera and timer
        self.cap = cv2.VideoCapture(0)  # Use 0 for the default camera
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # Setting up the main window
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle('Hand Detector App')
        self.show()

    def start_detection(self):
        if not self.timer.isActive():
            # Porneste camera si timer-ul odata cu apasarea butonului "Start Detection"
            self.cap = cv2.VideoCapture(0)  # Folosim 0 pentru camera default
            self.timer.start(30)  # Actualizare odata la 30 de milisecunde
            self.start_button.setText('Stop Detection')  # Actualizam textul butonului
            self.hands_photo_label.hide()  # Ascundem poza odata cu inceperea detectiei
        else:
            # Opreste camera si timer-ul odata cu apasarea butonului "Stop Detection"
            self.timer.stop()
            self.cap.release()
            self.start_button.setText('Start Detection')  # Actualizam textul butonului

    def update_frame(self):
        ret, img = self.cap.read()

        # Flipping the image for the model
        original_img = img.copy()  # Copying the original image
        img = cv2.flip(img, 1)

        # Converting to RGB for the model (model needs RGB image)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Passing the image to the model
        results = self.hands.process(rgb_img)

        # If there is any result (if any hand is detected)
        if results.multi_hand_landmarks:
            if len(results.multi_handedness) == 2:  # If two hands exist in the image
                x = 0
                y = 0
                for i in results.multi_handedness:
                    label = MessageToDict(i)['classification'][0]['label']
                    if label == 'Left':
                        x += 1
                    if label == 'Right':
                        y += 1
                if x == 1 and y == 1:
                    cv2.putText(original_img, 'Both Hands', (250, 56), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    if x == 2:
                        cv2.putText(original_img, 'Left Hands', (250, 56), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
                    else:
                        if y == 2:
                            cv2.putText(original_img, 'Right Hands', (250, 56), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                        (0, 255, 0), 2)
            else:  # If only one hand exists in the image
                for i in results.multi_handedness:
                    label = MessageToDict(i)['classification'][0]['label']
                    if label == 'Left':
                        cv2.putText(original_img, f'{label} Hand', (20, 56), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0),
                                    2)
                    if label == 'Right':
                        cv2.putText(original_img, f'{label} Hand', (460, 56), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0),
                                    2)

        # Converting the processed frame to QImage and updating the QLabel
        image = self.convert_frame_to_image(original_img)
        self.video_label.setPixmap(QPixmap.fromImage(image))

    def convert_frame_to_image(self, frame):
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return q_image.rgbSwapped()


def main():
    app = QApplication(sys.argv)
    window = HandDetectorApp()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
