from flask import Flask, render_template, request
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

numerical_cols = ['tempo', 'beats', 'chroma_stft', 'rmse',
                  'spectral_centroid', 'spectral_bandwidth', 'rolloff',
                  'zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5',
                  'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12',
                  'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19',
                  'mfcc20']
columns = numerical_cols


@app.route("/")
def index():
    return render_template("index.html", col=columns, enumerate=enumerate)


@app.route("/predict", methods=["GET", "POST"])
def new_prediction():
    tempo = request.form["tempo"]
    beats = request.form["beats"]
    chroma_stft = request.form["chroma_stft"]
    rmse = request.form["rmse"]
    spectral_centroid = request.form["spectral_centroid"]
    spectral_bandwidth = request.form["spectral_bandwidth"]
    rolloff = request.form["rolloff"]
    zero_crossing_rate = request.form["zero_crossing_rate"]
    mfcc1 = request.form["mfcc1"]
    mfcc2 = request.form["mfcc2"]
    mfcc3 = request.form["mfcc3"]
    mfcc4 = request.form["mfcc4"]
    mfcc5 = request.form["mfcc5"]
    mfcc6 = request.form["mfcc6"]
    mfcc7 = request.form["mfcc7"]
    mfcc8 = request.form["mfcc8"]
    mfcc9 = request.form["mfcc9"]
    mfcc10 = request.form["mfcc10"]
    mfcc11 = request.form["mfcc11"]
    mfcc12 = request.form["mfcc12"]
    mfcc13 = request.form["mfcc13"]
    mfcc14 = request.form["mfcc14"]
    mfcc15 = request.form["mfcc15"]
    mfcc16 = request.form["mfcc16"]
    mfcc17 = request.form["mfcc17"]
    mfcc18 = request.form["mfcc18"]
    mfcc19 = request.form["mfcc19"]
    mfcc20 = request.form["mfcc20"]

    data = CustomData(tempo, beats, chroma_stft, rmse, spectral_centroid, spectral_bandwidth,
                      rolloff, zero_crossing_rate, mfcc1, mfcc2, mfcc3, mfcc4, mfcc5, mfcc6, mfcc7, mfcc8,
                      mfcc9, mfcc10, mfcc11, mfcc12, mfcc13, mfcc14, mfcc15, mfcc16, mfcc17, mfcc18, mfcc19,
                      mfcc20)
    df = data.get_data_as_dataframe()
    model = PredictPipeline()
    prediction = model.predict(df)

    if prediction[0] == 1:
        prediction = "pop"
    elif prediction[0] == 2:
        prediction = "classical"
    elif prediction[0] == 3:
        prediction = "country"
    elif prediction[0] == 4:
        prediction = "disco"
    elif prediction[0] == 5:
        prediction = "hiphop"
    elif prediction[0] == 6:
        prediction = "jazz"
    elif prediction[0] == 7:
        prediction = "metal"
    elif prediction[0] == 8:
        prediction = "blues"
    elif prediction[0] == 9:
        prediction = "reggae"
    elif prediction[0] == 10:
        prediction = "rock"
    else:
        prediction = "Unknown genre"

    print(prediction)

    return render_template("result.html", predict=prediction)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
