from flask import Flask, request, render_template
from src.diabetesproject.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_datapoint():
    try:
        data = CustomData(
            Pregnancies=int(request.form.get("Pregnancies")),
            Glucose=float(request.form.get("Glucose")),
            BloodPressure=float(request.form.get("BloodPressure")),
            SkinThickness=float(request.form.get("SkinThickness")),
            Insulin=float(request.form.get("Insulin")),
            BMI=float(request.form.get("BMI")),
            DiabetesPedigreeFunction=float(request.form.get("DiabetesPedigreeFunction")),
            Age=int(request.form.get("Age"))
        )

        pred_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(pred_df)

        prediction = "Diabetic" if result[0] == 1 else "Not Diabetic"

        return render_template(
            "index.html",
            results=prediction
        )

    except Exception as e:
        return str(e)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)