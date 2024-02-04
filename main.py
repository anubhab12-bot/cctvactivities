import cv2
import numpy as np
import pyaudio
import wave
import threading
from flask import Flask, Response, render_template

app = Flask(__name__)

class ObjectTracker:
    def __init__(self, camera_index=0, output_file="cctv_footage.avi"):
        self.camera_index = camera_index
        self.output_file = output_file
        self.capture = cv2.VideoCapture(self.camera_index)
        self.tracker = cv2.TrackerMIL_create()  # Use a different tracker (MIL)
        self.is_recording = False
        self.is_audio_recording = False
        self.is_screen_recording = False
        self.video_writer = None
        self.audio_frames = []
        self.screen_recording_thread = None

    def start_recording(self):
        if self.output_file:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            frame_width = int(self.capture.get(3))
            frame_height = int(self.capture.get(4))
            self.video_writer = cv2.VideoWriter(self.output_file, fourcc, 15.0, (frame_width, frame_height))
            self.is_recording = True

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.video_writer.release()

    def start_audio_recording(self):
        self.is_audio_recording = True
        self.audio_thread = threading.Thread(target=self._record_audio)
        self.audio_thread.start()

    def stop_audio_recording(self):
        self.is_audio_recording = False
        self.audio_thread.join()

    def _record_audio(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=44100,
                        input=True,
                        frames_per_buffer=1024)

        while self.is_audio_recording:
            audio_data = stream.read(1024)
            self.audio_frames.append(audio_data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open("audio_record.wav", "wb")
        wf.setnchannels(1)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(self.audio_frames))
        wf.close()

    def start_screen_recording(self):
        self.is_screen_recording = True
        self.screen_recording_thread = threading.Thread(target=self._record_screen)
        self.screen_recording_thread.start()

    def stop_screen_recording(self):
        self.is_screen_recording = False
        if self.screen_recording_thread:
            self.screen_recording_thread.join()

    def _record_screen(self):
        while self.is_screen_recording:
            screen_frame = self.get_screen_frame()
            if self.is_recording:
                self.video_writer.write(screen_frame)

            _, encoded_frame = cv2.imencode(".jpg", screen_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_frame) + b'\r\n\r\n')

    def get_screen_frame(self):
        _, screen_frame = self.capture.read()  # You may need to adjust this based on your needs
        return screen_frame

    def track_object(self):
        ret, frame = self.capture.read()

        bbox = cv2.selectROI("Select Object to Track", frame, fromCenter=False, showCrosshair=True)
        self.tracker.init(frame, bbox)

        while True:
            ret, frame = self.capture.read()

            if ret:
                success, bbox = self.tracker.update(frame)

                if success:
                    x, y, w, h = [int(i) for i in bbox]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if self.is_recording:
                    self.video_writer.write(frame)

                cv2.imshow("Object Tracking", frame)

                _, encoded_frame = cv2.imencode(".jpg", frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_frame) + b'\r\n\r\n')

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.capture.release()
        cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(cctv.track_object(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/screen_feed')
def screen_feed():
    return Response(cctv._record_screen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    cctv = ObjectTracker(camera_index=0, output_file="cctv_footage.avi")
    cctv.start_recording()
    cctv.start_audio_recording()
    cctv.start_screen_recording()
    app.run(host='0.0.0.0', port=5000, debug=True)
