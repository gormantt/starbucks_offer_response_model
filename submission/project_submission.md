# Submission Information
Author: Tim Gorman </br>
Project: Starbucks Capstone Challenge

# Definition
This section will provide background on the project, a description of hte problem statement and the proposed solution.
## Project Overview
As part of the Udacity Machine Learning Engineer Course, Starbucks has provided a data science experiment for us to attempt.  This experiment is about understanding what the best offer is for each customer demographic that can be found in the Starbucks app at an invidualized, personalized level. The way that the challenge is presented leaves the door open for different approaches to this challenge. For example, I could build a machine learning model that predicts how much someone will spend based on demographics and offer type, I could build a model that predicts whether or not someone will respond to an offer, or I could decide not to build a machine learning model at all and instead define something like a rules engine.

Generally speaking, this project falls under marketing analytics which is the field of optimizing marketing campaigns for increased return on marketing investment. This interests me because marketing analytics is part of my everyday job at Huntington National Bank (HNB). In my role at HNB, I support model building and model deployment for marketing campaigns. Tackling this project will provide me with relevant experience to the problems I'm faced with every day at work. 

The data is provided in the AWS Starbucks Capstone Challenge work space. The data is contained in three files:
* portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
* profile.json - demographic data for each customer
* transcript.json - records for transactions, offers received, offers viewed, and offers completed

Here is the schema and explanation of each variable in the files:

* portfolio.json
    * id (string) - offer id
    * offer_type (string) - type of offer ie BOGO, discount, informational
    * difficulty (int) - minimum required spend to complete an offer
    * reward (int) - reward given for completing an offer
    * duration (int) - time for offer to be open, in days
    * channels (list of strings)
* profile.json
    * age (int) - age of the customer
    * became_member_on (int) - date when customer created an app account
    * gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
    * id (str) - customer id
    * income (float) - customer's income
* transcript.json
    * event (str) - record description (ie transaction, offer received, offer viewed, etc.)
    * person (str) - customer id
    * time (int) - time in hours since start of test. The data begins at time t=0
    * value - (dict of strings) - either an offer id or transaction amount depending on the record

To use this data I downloaded it from the provided workspace and uploaded it into my AWS Account for this section of th class.

## Brief Description of Problem
As described in the previous section, there are multiple ways to analyze the Starbucks dataset. For this project, I chose to build a model that predicts whether or not individuals will accept offers presented through the Starbucks app.

## Solution Statement
My solution was developed using Sagemaker Studio in the AWS account associated with this course. After appropriately analyzing and cleaning the data in a notebook,  I split the data into a training sets, validation sets, testing sets. I saved those data sets in S3 to and use them in Sagemaker Training jobs. I used a Sagemaker implementation of a package called [LightGBM](https://lightgbm.readthedocs.io/en/latest/index.html) to predict whether or not individuals will accept the offers presented through the Starbucks app. LightGBM is a gradient boosting framework that uses tree based learning algorithms. Tree based algorithms are well-suited for tabulaur data like in this Starbucks dataset and LightGBM in particular is advantageous because it is designed to be fast, have low memory usage, allow for parallel, distributed, and GPU learning, and handle large-scale data.  I trained and tested the model using sagemaker processing. I then compared this model output to to a logistic regression model and analyzed the results based on the metrics described below

## Metrics
I measured success based on the [AUC score](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc) (area under the ROC curve). This is an appropriate metric for classification problems that will give me a sense of the false positive and the true positive rate (extractable from the ROC Curve). Beyond that, AUC is a desirable metric for the following reasons: (1) AUC is scale-invariant (doesn't depend on absolute value of scores) and (2)
AUC is classification-threshold-invariant (measures quality of prediction independent of selected classification threshold). I also considered other metrics such as the [F1 Score](https://towardsdatascience.com/the-f1-score-bec2bbc38aa6) and accurracy.

# Analysis

## Data Exploration
My data exploration is all performed in the notebook "01_Exploratory_Data_Analysis". The important aspects of that exploration will be described below.

Before exploring the data the first thing As you can see in above in the [Project Overview Section](#Project-Overview) there are some similar columns between the various data sets but they are not exactly the same so I first normalized names across datasets so that customer ids are "customer_id" and offer_ids are "offer_id". With those column name updates, I will describe the three different datasets below.

### Portfolio
The portfolio dataset contains all 10 different offers presented in the Starbucks app and can be seen below.

![raw_portfolio.png](./images/raw_portfolio.png)

Each offer has a differing channels that in can be presented through with differing difficulties, rewards, and durations. The bogos come with the highest rewards and the discoutn comes with the highest difficulty. When we go to actually utilize this data in a model, we'll need to extract the individual channels out of the "channels" column and one-hot encode them.

### Profile
The profile dataset contains the relevant features of customer profiles as can be seen in the image below. "age", "customer_id", and "became_a_member_on" all have non-null values but "gender" and "income" both have null values that will need to be imputed before modeling.

![profile_stats.png](./images/profile_stats.png)

A snapshot of the profile dataset looks like the following.

![raw_profile.png](./images/raw_profile.png)
 
 By inspecting this sample of data we can see that the offer id is hashed like the customer id is from the portfolio dataset, and we can see that gender is presented as a categorical type. One column that stands out is "became_a_member_on", which can be reformed into something like a customer tenure, which I think will have a strong impact on whether or not an offer will be accepted.
 
 We can learn more about our numeric columns by ploting histograms of them, which ar shown below.
![profile_hist.png](./images/profile_hist.png)

"age" appears to have a fat tail on the distribution towards teh lower end with faster drop off on the higher end accompanied by a spike at the age of 118. We know that there aren't that many profiles with an age equal to 118 so we will be removing that from our dataset that gets fed into the model.

"became_a_member_on" gets grouped 

## Algorithms and Techniques
Algorithms and techniques used in the project are thoroughly discussed and properly justified based on the characteristics of the problem.

## Benchmark
Student clearly defines a benchmark result or threshold for comparing performances of solutions obtained.

# Methodology

## Data Preprocessing
All preprocessing steps have been clearly documented. Abnormalities or characteristics of the data or input that needed to be addressed have been corrected. If no data preprocessing is necessary, it has been clearly justified.

## Implementation
The process for which metrics, algorithms, and techniques were implemented with the given datasets or input data has been thoroughly documented. Complications that occurred during the coding process are discussed.

## Refinement
The process of improving upon the algorithms and techniques used is clearly documented. Both the initial and final solutions are reported, along with intermediate solutions, if necessary.

# Results

## Model Evaluation and Validation
The final model’s qualities—such as parameters—are evaluated in detail. Some type of analysis is used to validate the robustness of the model’s solution.

## Justification
The final results are compared to the benchmark result or threshold with some type of statistical analysis. Justification is made as to whether the final model and solution is significant enough to have adequately solved the problem.