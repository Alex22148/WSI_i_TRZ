from gesture_sensor import gesture, get_gesture_dict
from flask import Flask, jsonify

# Flask app
app = Flask(__name__)

gestures = gesture()

@app.route("/gesture", methods=["GET"])
def get_gesture_json():
    gesture_num = gestures.return_gesture()
    data = get_gesture_dict(gesture_num)
    if data != "nothing":
        print(data)
    """Returns absolute orientation in JSON format."""
    return jsonify(data)

if __name__ == "__main__":
    app.run(host="192.168.50.76", port=5000, debug=True)

