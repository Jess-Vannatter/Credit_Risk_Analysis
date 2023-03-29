# Credit_Risk_Analysis

## Overview of Analysis
  - The purpose of this analysis was to create various machine learning models paired with sampling approaches to help us determine if certain credit card applicants would either be High risk applicants or low risk applicants, meaning as in high risk they may default on their payments. By running various models, the goal was to predict which applications would be higher or lower risk based on several features that were portrayed as columns in a data set. These features were certain credit application related characteristics like Age, Loan amount, interest Rate, etc. with credit risk data being inherently imbalanced ( there are much more low risk applicants than high risk ), we would plan to test out 6 different models with six different sampling approaches to determine which would best address the imbalanced data and then best predict the outcomes of high to low risk applicants ( the actual outcome was already provided in one of the columns but we would use the data to train the model and compare the predicted outcome against the actual outcome).We would use imbalanced-learn and scikit-learn libraries to build these models, which you can see below with the predicted outcomes. Using the balanced accuracy score (because the initial data set, we used was imbalanced) along with the recall, and precision scores to determine the perceived outcomes. For this analysis the definition of recall was the proportion of actual positives that were identified correctly. While the definition of precision is the proportion of positive identifications that were correct.
  
## Results
  
  ### Naive Random Oversampling
  * This model returned a balanced accuracy score of 0.6612551982513499. Which in this case is not very accurate. The overall precision of the model was .99, with a 1.0 precision when predicting low risk applicants and a 0.01 precision score when predicting high risk applicants. But we had an over .66 recall score. Which means we were not very accurate when correctly predicting high risk applicants, which is the main purpose of the analysis. This model had an f1 score of .79, meaning the overall accuracy of the model to predict high/ low risk applicants was at about 79%.
  
 ![NRS CR](https://user-images.githubusercontent.com/117245167/227923874-c04379b0-ace1-4a6a-acc2-bfcc57d2d979.png)

   
  ### SMOTE Oversampling
  * This model returned a balanced accuracy score of 0.6475539275162315. Which in this case again, is not very accurate. The overall precision of the model was .99, with a 1.0 precision when predicting low risk applicants and a 0.01 precision score when predicting high risk applicants. We had an overall .64 recall score, which is even worse than the last model. this tells us our model had predicted a high amount of high risk applications to be low risk(or our model had a statistically significant amount of false negatives) Which means again, we were not very accurate when predicting high risk applicants. the f1 score for this model was .78, which is about the same as the previous model.
  
   ![SMOTE O CR](https://user-images.githubusercontent.com/117245167/227921881-70c87300-2118-46fa-80dd-0e9c7823a564.png)
   - 
    
  ### ClusterCentroids Undersampling
  * This model returned a balanced accuracy score of 0.5317926904944938. This is even less accurate than the last model accurate. The overall precision of the model was again .99 with a 1.0 precision when predicting low risk applicants and a 0.01 precision score when predicting high risk applicants. Meaning we were precise in predicting low risk applicants again. But we had an overall .39 recall score. Which means again, we were not very accurate when predicting high risk applicants. the f1 score for this model was .56 which is a significant regression in accuracy, which is not ideal. We would not want to use this model. 
  
   ![ClusterC U CR](https://user-images.githubusercontent.com/117245167/227921907-7e660095-05df-43f6-92ae-d940f8b42f73.png)
   - 
    
  ### SMOTEENN (Over and Under) Sampling
  * This model returned a balanced accuracy score of 0.647332509794478. This was a little better than the previous model and comparable to the first two. The overall precision of the model was .99, with a 1.0 precision when predicting low risk applicants and a 0.01 precision score when predicting high risk applicants. Meaning we were again precise in predicting low risk applicants again. But we had an overall .58 recall score. Which is not very accurate when predicting high/low risk applicants. The f1 score for this model was .73, which is better than the previous model, but comparable to the first two. 
  
   ![SMOTEENN U CR](https://user-images.githubusercontent.com/117245167/227921926-13d6c5c4-8084-412f-a10c-ba74b291ec22.png)
   - 
    
  ### Balanced Random Forest Classifier
  * This model returned a balanced accuracy score of 0.7885466545953005 . I would perceive this to be a drastic improvement, in comparison to the previous models. The overall precision of the model was .99, with a 1.0 precision when predicting low risk applicants and a 0.03 precision score when predicting high risk applicants. Meaning we were again, precise in predicting low risk applicants again. The recall for this model was also greatly increased with an overall recall of .87 and a.70 recall score when predicting high risk applicants and an .83 recall when predicting low risk. the f1 score for this model was at .93, which in comparison to the previous models is an impressive jump and by far the most accurate of the models so far. 
  
   ![BRFC CR](https://user-images.githubusercontent.com/117245167/227921950-a7e9bd9b-1baa-49e6-ac1a-1f27dc65e6ee.png)
   - 
    
  ### Easy Ensemble AdaBoost Classifier
  * This model returned a greatly increased balanced accuracy score of 0.9316600714093861 . I would perceive this to be a drastic improvement compared to the last model and the previous four before that one. The overall precision of the model was .99, with a 1.0 precision when predicting low risk applicants and a 0.09 precision score when predicting high risk applicants. So, the precision of this model when predicting high risk applicants also increased. The recall for this model was greatly increased with an overall recall of .94 and a .94 recall score when predicting high risk applicants and an .92 recall when predicting low risk. When comparing the six machine learning models, this model was clearly the most accurate and would provide the best option when predicting high/ low credit risk applicants with our data set. In addition, this model also had the highest f1 score of .97, which is the score that measures the overall accuracy. This is an impressive f1 score in relation to the other models.
  
  
  ![EEAC CR](https://user-images.githubusercontent.com/117245167/227921971-efddee99-9072-4044-8c9a-d649207a197b.png)
  - 
    
    
## Summary/ Recommendation
  - The scope of this particular analysis is credit risk analysis. So, with the goal to save our firm money, we would like to limit the number of high risk applicants receiving an approval, as they would be more likely to default on their bills/ credit. So ideally, we would like to have a higher recall score for the purpose of this analysis, as this would limit the number of actual high-risk applicants receiving an approved application. in other words, the higher the recall score the lower the risk we are providing our firm. Alternatively, precision is not as important for our analysis to an extent. Of course, we would like to make sure as many low-risk applicants get approved, as that also makes our firm money. But for the most part all these models were able to predict low risk applicants very well. So, the task should be to focus on correctly identifying high risk applicants. the final model, the Easy Ensemble AdaBoost Classifier model not only had an increased Precision score when predicting High risk applicants. But it also had the highest (by a large margin) recall score as well. Meaning this model and sampling approach was able to predict high risk applicants correctly at an impressive margin, while also limiting (in relation to the other models) the number low risk applicants from being denied. Ideally this is the model we would use going forward when predicting high/ low credit risk applicants when using similar features/ data points. 
