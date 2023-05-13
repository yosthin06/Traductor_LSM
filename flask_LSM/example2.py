import cv2
from flask import Flask, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def check_camera_access():
    camera = cv2.VideoCapture(0)
    success, frame = camera.read()
    del camera
    return success

def gen_frames():
    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():

    if check_camera_access():
        return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Error: Camera access not granted"

@app.route('/')
def index():
    return '''
        <!DOCTYPE html>
        <html>
            <head>
                <title>Live Video Feed</title>
            </head>
            <body>
                <h1>Live Video Feed</h1>
                <img src="/video_feed">
            </body>
        </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
