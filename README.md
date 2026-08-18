*# Diabetes Risk Prediction System*



*An end-to-end machine learning system that predicts diabetes risk from routine clinical measurements, returning a risk score, a High/Low classification, and rule-based lifestyle recommendations.*



*\*\*Live demo:\*\* https://diabetes-risk-prediction-lhuhw3pexafdm2t9b9esam.streamlit.app/*

*\*\*API:\*\* https://diabetes-risk-prediction-sis3.onrender.com*

*\*\*API docs (Swagger):\*\* https://diabetes-risk-prediction-sis3.onrender.com/docs*



*> Note: the backend runs on Render's free tier, which sleeps after inactivity. The first request after idle time may take 20-50 seconds while the server wakes up.*



*---*



*## Overview*



*Early detection of diabetes risk allows for lifestyle intervention before serious complications develop. This project trains and compares three classifiers on the PIMA Indians Diabetes dataset, serves the best-performing model through a REST API, and exposes it through a simple web form.*



*\*\*Input:\*\* Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age*

*\*\*Output:\*\* Risk percentage, High/Low risk label, and 2-4 lifestyle recommendations tied to the specific risk factors present*



*## Architecture*



*```*

*PIMA dataset (CSV)*

&#x20;     *|*

*Preprocessing (clean, scale, split)*

&#x20;     *|*

*Model training (Logistic Regression, Random Forest, XGBoost)*

&#x20;     *|*

*Saved model + scaler (joblib)*

&#x20;     *|*

*Backend API (FastAPI) ---> Recommendation engine (rule-based)*

&#x20;     *|*

*Frontend UI (Streamlit)*

&#x20;     *|*

*Deployment (Render + Streamlit Community Cloud)*

*```*



*- \*\*Model layer:\*\* `train\_model.py` trains three classifiers, compares them on accuracy/precision/recall/F1/ROC-AUC, and saves the best one along with its fitted scaler and feature order.*

*- \*\*Backend:\*\* `backend/app.py` is a FastAPI service that loads the saved model at startup and exposes a `/predict` endpoint with Pydantic input validation.*

*- \*\*Frontend:\*\* `frontend/app.py` is a Streamlit form that posts to the backend and renders the result.*



*## Dataset*



*\[PIMA Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) — 768 patient records, 8 input features, binary outcome (diabetic / not diabetic). Class distribution is imbalanced (\~65% negative / 35% positive).*



*\*\*Data quality note:\*\* Several columns (Glucose, BloodPressure, SkinThickness, Insulin, BMI) encode missing values as `0`, which is not physiologically possible for these measurements. These are converted to `NaN` and imputed with the column median before training. Insulin (48.7% missing) and Skin Thickness (29.6% missing) have substantial gaps — this is a known limitation of the dataset, not an artifact of this implementation.*



*## Model results*



*| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |*

*|---|---|---|---|---|---|*

*| Logistic Regression | 0.708 | 0.600 | 0.500 | 0.545 | 0.813 |*

*| Random Forest | 0.740 | 0.652 | 0.556 | 0.600 | 0.816 |*

*| \*\*XGBoost (selected)\*\* | \*\*0.760\*\* | \*\*0.673\*\* | \*\*0.611\*\* | \*\*0.641\*\* | \*\*0.827\*\* |*



*XGBoost was selected by F1 score rather than accuracy, since a model that always predicted "no diabetes" would score \~65% accuracy on this class distribution while being clinically useless.*



*\*\*Limitation:\*\* recall of 0.61 for the diabetic class means roughly 39% of true positive cases are missed on the test set. A production screening tool would likely tune the decision threshold to favor recall over precision, since a missed diagnosis is generally costlier than a false alarm.*



*## Tech stack*



*| Layer | Tools |*

*|---|---|*

*| Data \& modelling | Python, pandas, NumPy, scikit-learn, XGBoost, joblib |*

*| Backend | FastAPI, Pydantic, Uvicorn |*

*| Frontend | Streamlit, requests |*

*| Deployment | Render (backend), Streamlit Community Cloud (frontend) |*



*## Project structure*



*```*

*diabetes-risk-prediction/*

*├── backend/*

*│   └── app.py              # FastAPI service*

*├── frontend/*

*│   └── app.py               # Streamlit UI*

*├── data/*

*│   └── diabetes.csv         # PIMA dataset*

*├── models/*

*│   ├── diabetes\_model.pkl   # trained XGBoost model*

*│   ├── scaler.pkl           # fitted StandardScaler*

*│   └── feature\_order.pkl    # feature ordering used at inference*

*├── docs/*

*│   ├── eda\_plots.png        # EDA visualizations*

*│   └── model\_metrics.csv    # model comparison table*

*├── train\_model.py           # EDA + training script*

*├── render.yaml               # Render deployment config*

*├── requirements.txt*

*└── README.md*

*```*



*## Running locally*



*### 1. Clone and set up the environment*

*```bash*

*git clone https://github.com/PRANEETHJARPLA/diabetes-risk-prediction.git*

*cd diabetes-risk-prediction*

*python -m venv venv*

*venv\\Scripts\\Activate.ps1        # Windows*

*# source venv/bin/activate       # macOS/Linux*

*pip install -r requirements.txt*

*```*



*### 2. Train the model (optional — pretrained artifacts are already in `models/`)*

*```bash*

*python train\_model.py*

*```*



*### 3. Run the backend*

*```bash*

*uvicorn backend.app:app --reload*

*```*

*API available at `http://127.0.0.1:8000`, interactive docs at `http://127.0.0.1:8000/docs`.*



*### 4. Run the frontend*

*In a separate terminal (with the venv activated):*

*```bash*

*streamlit run frontend/app.py*

*```*

*UI available at `http://localhost:8501`.*



*> Note: `frontend/app.py` is currently configured to call the deployed Render backend. To test against a local backend instead, change `API\_URL` in `frontend/app.py` to `http://127.0.0.1:8000/predict`.*



*## API reference*



*\*\*POST\*\* `/predict`*



*Request body:*

*```json*

*{*

&#x20; *"Pregnancies": 1,*

&#x20; *"Glucose": 85,*

&#x20; *"BloodPressure": 66,*

&#x20; *"SkinThickness": 29,*

&#x20; *"Insulin": 0,*

&#x20; *"BMI": 26.6,*

&#x20; *"DiabetesPedigreeFunction": 0.351,*

&#x20; *"Age": 31*

*}*

*```*



*Response:*

*```json*

*{*

&#x20; *"risk\_percentage": 1.0,*

&#x20; *"risk\_label": "Low",*

&#x20; *"recommendations": \[*

&#x20;   *"Weight management: aim for gradual, sustainable weight loss",*

&#x20;   *"Regular exercise: at least 150 minutes/week of moderate activity"*

&#x20; *]*

*}*

*```*



*## Roadmap*



*- \[ ] Threshold tuning to improve recall for the diabetic class*

*- \[ ] SHAP-based explainability per prediction*

*- \[ ] Heart disease risk module*

*- \[ ] Mobile-friendly UI*



*## Author*



*Praneeth Jarpla — Instrumentation \& Control Engineering, NIT Rourkela*

*GitHub: \[@PRANEETHJARPLA](https://github.com/PRANEETHJARPLA)*

