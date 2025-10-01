import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv("data/sample_data.csv")

X = df.drop(columns=["patientid","target"])
y = df["target"]

# Select columns
num_cols = ["age","resting_blood_pressure","serum_cholesterol",
            "maximum_heart_rate_achieved","oldpeakst"]
cat_cols = ["gender","chestpain","fasting_blood_sugar",
            "resting_electrocardiogram_results","exercise_induced_angina",
            "slope_of_the_peak_exercise_st_segment"]

# Preprocessing
num_transform = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_transform = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("num", num_transform, num_cols),
    ("cat", cat_transform, cat_cols)
])

# Model
pipe = Pipeline([
    ("preproc", preprocessor),
    ("clf", LogisticRegression(max_iter=1000))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

pipe.fit(X_train, y_train)

print("Training complete. Saving model...")
joblib.dump(pipe, "model.joblib")
