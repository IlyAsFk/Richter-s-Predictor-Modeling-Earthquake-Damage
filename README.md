# Implementation and Reproducibility

**Project goal:** Predict the damage level of buildings after the 2015 Gorkha (Nepal) earthquake. The target is the damage grade per building (multiclass classification). This repository contains data files, trained models, model outputs (submission files), and an R notebook implementing the end-to-end pipeline.

**Notebook (reproducible workflow)**
- `Richter-s-predictor-group-15.ipynb`: R notebook (runs with an R kernel) that contains the full data loading, exploratory data analysis (EDA), preprocessing, model training, evaluation, and prediction steps. Run this notebook to reproduce results and regenerate models/submissions.

**Data files (input)**
- `train_values.csv`: Building attributes/features for training.
- `train_labels.csv`: Training labels (building id + damage grade). Use to train and evaluate models.
- `test_values.csv`: Unlabeled test set to produce final predictions for submission.
- `submission_format.csv`: Example submission template (ID column(s) + damage grade column placeholder).

**You can save trained models and artifacts**
- `bayes_train.rds`: Naive Bayes trained model (saved with `saveRDS`).
- `dt_model.rds`: Decision Tree trained model.
- `rf_rtree300_importance.rds`: Random Forest (300 trees) used to compute feature importance.
- `rf_rtree300_train.rds`: Random Forest (300 trees) final model used for our best submission.
- `rf_model_3_rtree500.rds`: Random Forest variant (500 trees) explored during tuning/hyperparameter search.
- `dnn_model/`: Directory containing the trained deep neural network model (TensorFlow/Keras saved model files: `saved_model.pb`, `variables/`, `assets/`, plus `fingerprint.pb` and `keras_metadata.pb`).

**Submission files have to be saved as csv files**
- `by_model_submission_format.csv` — predictions from the Naive Bayes model.
- `dnn_model_submission_format.csv` — predictions from the DNN model.
- `dt_model_submission_format.csv` — predictions from the Decision Tree model.
- `rf_model_submission_format.csv` — predictions from the Random Forest final model.

Each CSV follows the `submission_format.csv` template and was produced by running the notebook's prediction cells with the corresponding saved model.

**Exploratory Analysis & Preprocessing (what was done)**
- Exploratory Data Analysis (EDA): distribution checks, missing value inspection, and a correlation matrix (saved as `correlation_plot.png`) to inspect multicollinearity and relationships between features and the damage grade.
- Data cleaning: handling of missing values, consistent treatment of categorical variables (factor encoding), and alignment of train/test feature columns.
- Feature engineering: creation/transformation of derived features where helpful (documented in the notebook), and selection/inspection of top features using Random Forest importance.
- Data splitting: training/validation splits and cross-validation were used during model development (see notebook for exact folds and seeds).

**Models trained and tuning**
- Naive Bayes: simple baseline model.
- Decision Tree (rpart or similar): interpretable baseline.
- Random Forests (multiple variants): used extensively — one run with 300 trees was used to compute feature importance (`rf_rtree300_importance.rds`), another 300-tree model was used as the final predictor (`rf_rtree300_train.rds`). A 500-tree variant (`rf_model_3_rtree500.rds`) was used during parameter exploration.
- Deep Neural Network (DNN): implemented in TensorFlow/Keras (saved under `dnn_model`).
- Hyperparameter tuning: grid/random search and manual tuning were performed for the tree-based models; details and exact parameter grids are available in the notebook.

**Model selection and evaluation**
- Evaluation metrics: classification accuracy and class-weighted metrics (precision/recall/F1) were computed during validation. The selected model for final submission (`rf_rtree300_train.rds`) produced the best held-out validation performance among tested candidates.
- Feature importance from Random Forest guided feature-reduction and interpretation.

**How to reproduce predictions (quick commands)**
Run the notebook (`Richter-s-predictor-group-15.ipynb`) top-to-bottom in Jupyter (with an R kernel) or open it in RStudio and run the cells. Alternatively, the shortest programmatic way to produce predictions from a saved R model:

```r
# Minimal example (R) to load a saved model and predict on test set
library(readr)
# adjust package names according to what you used (data.table, dplyr, etc.)

# Load test data
test <- read_csv("test_values.csv")
# Load saved Random Forest model (example)
rf <- readRDS("rf_rtree300_train.rds")
# Make predictions (may require pre-processing same as training)
preds <- predict(rf, newdata = test, type = "response")
# Prepare submission (ensure column names match template)
sub <- read_csv("submission_format.csv")
sub$damage_grade <- as.integer(preds)
write_csv(sub, "rf_model_submission_format.csv")
```

For the DNN model, predictions were generated using a Keras/TensorFlow pipeline saved under `dnn_model/`. Load with `keras::load_model_tf("dnn_model")` (or equivalent) and apply the same preprocessing pipeline.

**Required packages (R, typical list)**
- `tidyverse` (or `readr`, `dplyr`, `ggplot2`),
- `data.table`,
- `randomForest`,
- `rpart`,
- `e1071` (for Naive Bayes),
- `caret` (for tuning/validation),
- `keras`/`tensorflow` (for DNN),
- `yardstick` or `MLmetrics` (for evaluation).

Install with (R):

```r
install.packages(c("tidyverse","data.table","randomForest","rpart","e1071","caret","MLmetrics"))
# For keras/tensorflow follow https://tensorflow.rstudio.com/install/
```

**Files summary**
- Code & notebook: `Richter-s-predictor-group-15.ipynb` (run this to reproduce everything).
- Raw data: `train_values.csv`, `train_labels.csv`, `test_values.csv`.
- Models/artifacts: `*.rds`, `dnn_model/`.
- Submissions file: `*_submission_format.csv`.
- Plots: `correlation_plot.png`.

**Next steps / suggestions**
- Run the notebook end-to-end to re-generate models and submissions and validate scores locally.
- Consider ensembling multiple model predictions (stacking/voting) to improve leaderboard performance.

