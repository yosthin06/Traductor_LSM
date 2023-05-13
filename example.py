import os
import cv2

from flask import Flask, render_template, Response, request

app = Flask(__name__)

def hello_world():
    
    cam = cv2.VideoCapture(0)
    
    while True:
        check, frame = cam.read()        

        ret, buffer = cv2.imencode('.jpg', frame)
                
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    
	#return "hello world my name is Yosthin"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(hello_world(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
	app.run(debug=True,host="0.0.0.0",port=int(os.environ.get("PORT",8080)))