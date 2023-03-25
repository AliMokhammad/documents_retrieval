import init
import json
import threading
from datetime import datetime
from flask import Flask, render_template, request
from constants import ui, server_port, models_dir
from utils import * 

app = Flask(__name__, template_folder="ui")

@app.route("/")
def home():
    return render_template(ui["home"])


@app.route("/preprocess", methods=["GET", "POST"])
def preprocess():
    if request.method != "POST":
        return render_template(ui["preprocess"])

    dataset = validate_req_dataset(request.files)
    
    if not dataset["is_valid"]: return dataset["msg"]

    col_key = request.args.get("col_key")

    if not col_key: return "No key was selected read text"

    t = threading.Thread(target=preprocess_dataset, args=(dataset["file"],col_key,))
    t.start()
    # Return a message indicating that the task is in progress
    return "Dataset is being processed. Keep tracking of console logs."


@app.route("/train", methods=["GET", "POST"])
def train():
    if request.method != "POST":
        return render_template(ui["train"])
    
    now = datetime.now()
    model_name = now.strftime("%Y_%m_%d_T_%H_%M_%S")
    print("Model Name:", model_name)
    
    t = threading.Thread(target=train_new_model, args=(model_name,))
    t.start()

    return f"Model \"{model_name}\" is being trained, and will be saved in \"{models_dir}\" directory. Keep tracking of console logs."


@app.route("/test", methods=["GET", "POST"])
def test():
    if request.method != "POST":
        return render_template(ui["test"])
    res = test_model(request.json)
    return json.dumps(res)
     


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=server_port)
