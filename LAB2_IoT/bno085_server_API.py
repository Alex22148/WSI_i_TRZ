from flask import Flask, jsonify
import board
import busio
from bno085_reader import get_orientation

# Flask app
app = Flask(__name__)


@app.route("/orientation", methods=["GET"])
def get_orientation_json():
    """Returns absolute orientation in JSON format."""
    data = get_orientation()
    return jsonify(data)

if __name__ == "__main__":
    app.run(host="192.168.50.76", port=5000, debug=True)
