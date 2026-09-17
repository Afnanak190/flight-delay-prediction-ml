# Flight Delay Prediction & Root Cause Analysis

Predicting flight delays using scheduling data, with statistical validation of key drivers.

Tools: Python · pandas · scikit-learn · scipy (statistics)

What This Project Does

Built a machine learning pipeline to predict whether a US domestic flight will be delayed (arrival delay > 15 minutes), using only information available at the time of scheduling — carrier, route, date, and scheduled times. Also used statistical testing to validate which factors actually matter.

Dataset
Source: Airline Delay Causes (Kaggle)
Size: ~1.9 million flight records after cleaning
Note: this dataset only contains flights that experienced some delay, so the "on-time" class here is underrepresented compared to real-world flight traffic — something to keep in mind when interpreting the results.
Approach

1. Data Cleaning (Python)

Dropped delay-cause columns (CarrierDelay, WeatherDelay, etc.) since these are only populated after a delay occurs — using them would leak the answer into the model.
Removed rows with missing arrival/departure delay values (mostly cancelled or diverted flights).

2. Feature Selection Only used features that would be known before a flight departs — Month, Day, Day of Week, scheduled departure time, carrier, origin, destination, distance, and scheduled flight duration. Also engineered two extra features: departure hour and a winter-season flag.

3. Modeling

Built a binary target (Is_Delayed) based on arrival delay > 15 minutes
Trained and compared two models: Logistic Regression (baseline) and Random Forest
Random Forest performed slightly better after feature engineering

4. Statistical Validation

Ran a Chi-square test to check whether airline carrier is genuinely associated with delay likelihood
Results
Model	Accuracy
Logistic Regression	62.9%
Random Forest (baseline)	64.6%
Random Forest (with engineered features)	65.0%

Honest take on accuracy: 65% is a modest result, and that's an important finding in itself. Using only scheduling information (date, route, carrier) has limited predictive power for something as dynamic as flight delays. Real-world factors that matter a lot — live weather, air traffic congestion, and whether the incoming aircraft is already running late — simply aren't available in this dataset. This is consistent with published research on flight delay prediction, where schedule-only models typically land in a similar range.

Feature Importance
Feature	Importance
Carrier (airline)	26.9%
Scheduled departure time	9.9%
Month	9.8%
Day of month	9.4%
Distance	9.1%
Destination	7.8%
Scheduled elapsed time	6.9%
Origin	6.7%
Departure hour	5.6%
Winter flag	4.0%
Day of week	3.8%

Which airline (carrier) turned out to be the single strongest predictor — more than double the next most important feature.

Statistical Test

Ran a Chi-square test of independence between Carrier and Delay:

Chi-square statistic: 36,169.2
p-value: < 0.001

This confirms that the relationship between carrier and delay isn't due to chance — some airlines are statistically more prone to delays than others, backing up what the model's feature importance already suggested.

Recommendations
Prioritize on-time performance improvements for the carriers showing the highest delay association
Since schedule-only data has limited predictive power, incorporating live weather and air traffic data would meaningfully improve future delay-prediction models
Files in This Repository
File	Contents
flight_delay_prediction.ipynb	Full Python notebook — cleaning, modeling, evaluation, statistical test

Note: The trained model file (.pkl) is not included due to its large file size. It can be reproduced by running the notebook end-to-end.

Skills Demonstrated

Python, pandas, scikit-learn (Logistic Regression, Random Forest), feature engineering, statistical hypothesis testing (Chi-square), model evaluation, honest results interpretation
