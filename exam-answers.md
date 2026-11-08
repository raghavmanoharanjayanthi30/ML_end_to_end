# Technical Exam Answers

## Section A: ML Foundations (6 points)

### Q1 (3 pts)
Define overfitting, underfitting, and the bias-variance tradeoff. For each, describe one concrete signal you would see in your training metrics or validation curves that would tell you you're in that regime. Use specific terminology from your scikit-learn build where it applies.

**Answer:**
Overfitting:
Low bias, high variance
Great on training, poor on validation/test
Signal: Training loss keeps decreasing, but validation loss starts increasing (the gap widens)

Underfitting:
High bias, low variance (not high variance)
Model is too simple to capture patterns
Signal: Both training and validation error are high (and similar to each other)

Bias-Variance Tradeoff:

Simple models → high bias, low variance (underfitting)
Complex models → low bias, high variance (overfitting)
Goal: find the sweet spot in between

In my build measuring ROC-AUC
In GridSearchCV, if the best model has much higher mean_train_score than mean_test_score, this indicates overfitting; if both scores are low, the underfitting

### Q2 (3 pts)
You're training a binary classifier on a 5% positive class. Walk through the metrics you would prioritize, in order, and explain why accuracy is not on your list. Name a specific scenario where you would optimize for recall over precision, and one where you would optimize for the reverse.

**Answer:**
metrics i would optimize for: precision, recall, f1 score (combination of precision, recall)
precision - out of how many model predicts to be positive class, how many are actually positive class
recall - what % of actual positives does the model catch
f1 - 2 * precision * recall (precision + recall) -> good balance between both
accuracy - not important because in imbalanced class, if you just predict all to be the negative class you get 95% accuracy but its a useless model that never predicts the positive class
AUC-ROC: threshold-independent and good for imbalanced classes.
Precision@k / Recall@k 

recall over precision - fraud detection -> false negative worse than false positive
precision over recall - model that decides whether to give a loan (positive) or not (negative) -> giving loan to wrong person (false positive) worse than missing out on giving to right person (false negative)


---

## Section B: Data Engineering & Quality (6 points)

### Q3 (3 pts)
Compare ETL and ELT. For each: what kind of data infrastructure assumes which pattern, what are the tradeoffs, and where does the line blur for ML workloads specifically? Give a concrete scenario where you would pick each.

**Answer:**
ETL (Extract, Transform, Load)

Transform happens on a separate processing server before loading to destination
Infrastructure: Traditional data warehouses (Oracle, SQL Server), on-prem systems
Tradeoff: Cleaner data lands in destination, but slower and requires upfront schema design

ELT (Extract, Load, Transform)

Load raw data first, transform inside the destination using its compute power
Infrastructure: Modern cloud warehouses (Snowflake, BigQuery, Databricks, Redshift) that have cheap storage + powerful compute
Tradeoff: Faster ingestion, more flexible (transform later as needs change), but raw data takes up space

ML blur because:
need ELT to get raw data in fast
then ETL style transformations to transform the data at training/inference time

When pick ETL
- have traditional data warehouse with limited computed/storage and data in a particular format; for example, bank transaction data into oracle (expensive storage, rigid schema)

WHen pick ELT
- have modern cloud warehouse with better compute, cheaper storage, different data formats accepted, flexibility to transform data in different ways later
ex: raw data load then data scientists can use all raw data and features to prepare data for modeling use as needed (different transformations for different experiments)

### Q4 (3 pts)
You receive a daily 200MB CSV from an upstream system. Some days it has missing columns, some days a column changes type, sometimes the file is 50MB and you don't notice. Walk through how you would catch these failures BEFORE they corrupt your training run.

**Answer:**
Have validation checks before training like the one built in the Great Expectations Suite

example checks
- all columns present
- no unexpected columns
- column data types align; values are valid - ex age integer and between 0 and 99
- monitor file size, number of rows
- missingness check
- uniqueness check

when validation fails - abort the pipeline with a clear error message, and no training
validation step has to pass before we get to training

---

## Section C: Experiment Tracking & Versioning (6 points)

### Q5 (3 pts)
Why do ML teams version data with DVC instead of just committing it to Git? Explain at least three problems DVC solves. Then describe a specific scenario where DVC alone isn't sufficient and you'd reach for something else (DataLake, Delta Lake, Iceberg, etc.).

**Answer:**
DVC addresses many problems
- File size: Git can't handle large files DVC stores a small hash in Git, actual data elsewhere.
- Storage efficiency: Git stores full copies of every version. DVC deduplicates - if only 1% of data changes, it doesn't re-store the whole file
- Reproducibility: Links specific data versions to specific code commits. You can checkout an old commit and dvc checkout to get the exact data that was used for that experiment

DVC is good if the data is static and one person updates at a time.

DVC bad if
- data is not static; can't work with streaming data
- concurent updates from multiple users; Delta Lake/Iceberg handle concurrent writes with ACID transactions
- delta lake supports sql queries on data; dvc only stores files

### Q6 (3 pts)
Walk through what MLflow tracks in a typical run and what it does NOT track that you still need to capture somewhere else. Then describe how you would use the MLflow model registry to manage promotion from staging to production. Be specific about the gates.

**Answer:**
ML flow tracks stuff like
- model hyperparameters in different experiments
- model performance metrics like AUC, precision, recall, loss, accuracy etc
- model artifacts like model files, feature importance plots etc
- source code version (git commit hash)
- experiment metadata - time of experiment, who ran it etc

but it doesn't track stuff like
- data version changes, need DVC
- infrastructure: what machine, what GPU used to run experiment etc
- production metrics like latency
- feature engineering changes -> if you change feature engineering but don't commit, it doesn't know

for staging to production
- train multiple models which are all logged to ML Flow
- register the best candidate - move to staging in model registry
- gates before production - cross validation performance meets threshold, acceptable latency and throughput, no data drift etc
- promote to production

Challenger pattern (you nailed this):
- New models run shadow inference alongside Champion
- Compare real world performance over 30-60 days
- If Challenger outperforms - promote to new Champion

---

## Section D: Deployment & CI/CD (6 points)

### Q7 (3 pts)
You have a trained scikit-learn model. Compare three deployment patterns: batch inference (run nightly, write to a table), real-time API (FastAPI endpoint behind a load balancer), and embedded (model file shipped with a client app). For each: what is the right business scenario, what are the operational risks, and what monitoring is hardest?

**Answer:**
Batch
- for not time sensitive predictions, ex: churn prediction (this project), run bi-weely or montly
- risks: data drift is detected only every 2 weeks or monthly, stale predictions

Real time
- for instant predicitons, ex fraud detection, credit approval
- risks: high latency, scaling under load, dependency failures, server downtime
- monitoring errors, latency, throughput for such high volume can be difficult

embedded
- offline use = he model runs on the user's device, not on your server. example: iPhone keyboard suggestions - the model predicting your next word runs on your phone, not Apple's server
- can't update model easily, need to roll in the model update through an app upgrade; different devices = different behavior, no visibility into usage



### Q8 (3 pts)
Your CI pipeline currently runs unit tests on every PR. What additional checks would you add for an ML project that you wouldn't add for a typical software project? Be specific, not just "run the training" but what artifacts you'd test, what you'd block merges on, and what you'd leave to manual review.

**Answer:**
- data validation checks: great expectations suite
- feature data checks: no blocked features in pipeline
- Schema checks - Feature pipeline input/output shapes match expected
- model loads and predictions: model loads, and can generate valid predictions ex: prediciton probablitiies between 0 and 1


---

## Section E: Monitoring & Production (6 points)

### Q9 (3 pts)
Define data drift, concept drift, and prediction drift. For each, give one concrete signal you would compute and the threshold you would alert on. Explain why a system with no drift detection becomes silently broken faster than a typical software system.

**Answer:**
Data drift
KS test (numeric) or Chi-square (categorical) per feature
Alert if p < 0.01 for any feature

Prediction drift
KS test on predicted probabilities, or mean prediction shift
Alert if mean shifts >10% or KS p < 0.01

Concept drift
AUC on labeled data (when available)
Alert if performance drops >5%

data drift
data disitbruiton changes
fraud detetcion model - global model - originally 80% of trnaactions USA based, but now only 20%, more european trnaactions

prediciton drift
prediction distirbution changes over time ex: month 1 60% positve month 3 53%, month 4, 35%

concept drift
relationship betwene input and ouput changes
ex: fraud detection model
over time fraud behavior changes

ML models fail silently, they keep producing predictions, just wrong ones. No error is thrown. 

### Q10 (3 pts)
Your model has been in production for 6 months. Performance has degraded 8 percentage points on AUC and no one knows why. Walk through your debugging process step by step. What do you check first, second, third? When would you retrain vs. roll back vs. take the model down?

**Answer:**
Debugging process (in order):

Data pipeline check - Is data even flowing correctly
Data drift - Compare current input distributions to training data. Which features shifted? 
Subgroup analysis - Is degradation uniform or concentrated? Maybe only one state or plan_type is broken.
Upstream changes - Did a data source change? New vendor? new vendor - different units of data ex: temperature in celisus instead of fahrenehit (model sees completely different range of values for temperature as an input feature)
Concept drift - Has the relationship between features and outcome changed?
Label quality - Are the ground truth labels still accurate? Did the definition of "churn" change?

Model decisions
If gradual over 6 months, likely drift, retrain on recent data
If sudden drop, likely upstream data issue or bad deployment, rollback + investigate
If model is now predicting unfairly against a protected group, take down immediately
