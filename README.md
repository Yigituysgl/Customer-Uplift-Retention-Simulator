# Customer-Uplift-Retention-Simulator
This project builds an end-to-end marketing uplift modeling system that identifies which customers are most likely to respond positively to retention campaigns.
The workflow includes:
Data preprocessing
RFM customer segmentation
Churn probability prediction using ML
Synthetic marketing campaign simulation (treatment vs control)
Uplift modeling with two-model Gradient Boosting
Uplift scoring for every customer
Target customer selection (Top uplift deciles)
Interactive Streamlit Dashboard
This project is designed for real-world marketing teams to optimize retention campaigns and reduce churn.

1. Feature Engineering: RFM Segmentation
From transactional records, the following were computed:
Recency — days since last purchase
Frequency — number of purchases
Monetary — total spend

2-Churn Probability Model
A Gradient Boosting Classifier predicts each customer’s probability of churn:
Inputs: Recency, Frequency, Monetary
Output: Churn_Prob (0.0–1.0)
This probability is later used to simulate how likely a customer is to respond to a campaign.

3-Synthetic Treatment–Control Campaign Simulation
Because the dataset has no real marketing campaign labels, a synthetic uplift experiment is created:
Customers with higher churn risk are assigned Treatment = 1
Low-risk customers form the Control group
A response outcome is simulated based on churn probability and treatment
This gives realistic T/C data suitable for uplift modeling.

4-Uplift Modeling (Two-Model Approach)
Two Gradient Boosting models are trained:
Model A — trained on customers who received treatment
Model B — trained on customers in the control group

5-Uplift Scoring
A new column is added:
Uplift_Score — value between approx. –0.8 to +0.9
Higher score = more likely to respond positively to the campaign.

6-Streamlit Dashboard
An interactive dashboard visualizes:
Total customers
Selected target customers
Average uplift
Uplift distribution
Filters:
Risk segment
Minimum uplift threshold

<img width="1895" height="943" alt="image" src="https://github.com/user-attachments/assets/74ad2dad-f891-432f-9309-31a4ad0504dc" />
<img width="1665" height="943" alt="image" src="https://github.com/user-attachments/assets/b275a7b3-c895-450b-8ec4-468d41b88ea8" />
<img width="1760" height="837" alt="image" src="https://github.com/user-attachments/assets/b9850d6c-c303-4658-acd0-7c3c8a53e84e" />




