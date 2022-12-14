# Predict Future Sale

* This repository holds an attempt to predict total sales for every product and store in the next month from kaggle challenge 'Predict Future Sales.'
* link - https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/overview/description

## Overview


 * **Definition of the tasks / challenge:**  The task is to forecast the total amount of products sold in every shop for the next month with daily historical sales data provided.
 
 * **Your approach:** The approach in this repository formulates the problem by using deep learning models and gradient boost framework with the time series data as the input. The metric considered is root mean squared error. We compared the performance of 3 different network architectures.
 
 * **Summary of the performance achieved:** Our best model was able to predict the product sale for the shop and item for the next month with the metric(rmse) of 1.11229.

## Summary of Workdone

### Data

* Data:
  * Files:
    * File descriptions:
      *sales_train.csv - the training set. Daily historical data from January 2013 to October 2015.
      *test.csv - the test set. You need to forecast the sales for these shops and products for November 2015.
      *sample_submission.csv - a sample submission file in the correct format.
      *items.csv - supplemental information about the items/products.
      *item_categories.csv  - supplemental information about the items categories.
      *shops.csv- supplemental information about the shops.
      
  * Type:
    * Input: CSV file of following features:
      * ID - an Id that represents a (Shop, Item) tuple within the test set
      * shop_id - unique identifier of a shop
      * item_id - unique identifier of a product
      * item_category_id - unique identifier of item category
      * item_cnt_day - number of products sold. Predicting a monthly amount of this measure
      * item_price - current price of an item
      * date - date in format dd/mm/yyyy
      * date_block_num - a consecutive month number, used for convenience. January 2013 is 0, February 2013   is 1,..., October 2015 is 33
      * item_name - name of item
      * shop_name - name of shop
      * item_category_name - name of item category 
      
    * Output: item_cnt_month - item sale for the next month.
    
  * Size: 101.61 MB
  
  * Instances (Train, Test, Validation Split): data points: 3150047. 2882334 shop_id and items for training, 214200 for testing, 53513 for validation

### Preprocessing / Clean up

* Removed outliers for item_price and item_cnt_day
* Encoded shop_name to city_code, item_category_name to type_code and subtype_code
* Added the following features: 
     * revenue - revenue obtained for the particular datapoint                      
     * item_cnt_month - no. of items sold in a month                                  
     * date_avg_item_cnt - average no. of items sold per month for each month            
     * date_item_avg_item_cnt - mean no. of items sold per month for each month and item_id   
     * date_shop_avg_item_cnt - mean no. of items sold per month for each month and shop_id       
     * date_cat_avg_item_cnt - mean no. of items sold per month for each month and item_category_id        
     * date_shop_cat_avg_item_cnt - mean no. of items sold per month for each month and shop_id and item_category_id 
     * date_shop_type_avg_item_cnt -  mean no. of items sold per month for each month and shop_id and type_code  
     * date_shop_subtype_avg_item_cnt - mean no. of items sold per month for each month and shop_id and subtype_code
     * date_city_avg_item_cnt - mean no. of items sold per month for each month and city_code    
     * date_item_city_avg_item_cnt - mean no. of items sold per month for each month and item_id and city_code 
     * date_type_avg_item_cnt - mean no. of items sold per month for each month and type_code       
     * date_subtype_avg_item_cnt - mean no. of items sold per month for each month and subtype_code   
     * item_avg_item_price - mean of item_price for each item_id         
     * date_item_avg_item_price - mean of item_price for each month and item_id
     * date_shop_revenue - sum of revenue for each month and shop_id  
     * shop_avg_revenue - mean of revenue per month and shop for each shop_id              
     * delta_revenue - how close the date_shop_revenue is to shop_avg_revenue for each shop           
     * month - month number for each date_block_num
     
### Data Visualization

Histogram plots were used data visualization in this project.

Figure 1:

<img width="776" alt="Screenshot 2022-12-13 162543" src="https://user-images.githubusercontent.com/89792366/207458087-9c015e90-055a-46a4-802a-46b9121dee37.png">

Figure 1: Large number of items are sold during the end of each year, but the overall trend for the product sale is decreasing thorughout the timeline. Thus, variables like month, item price for all months, items sold in cities for all months, etc. were added to training dataset.


Figure 2:

<img width="767" alt="Screenshot 2022-12-13 162714" src="https://user-images.githubusercontent.com/89792366/207458271-dc88c351-204a-4794-81e3-e63b06e7be0b.png">

Figure 2: The 31st shop sold the largest number of items (>300000). 0-24 shops sold 0-100000 items. 25-30 sold a large number of items as well. It shows that the shops between 25-31 are extremely good in selling products, while shops from 0-4,8-11,32-34,36,39-40 are selling less products. Thus variables realted to shops like revenue per shop, etc. were considered.


Figure 3:

<img width="771" alt="Screenshot 2022-12-13 162827" src="https://user-images.githubusercontent.com/89792366/207458440-d36d2078-b6e9-4e98-b238-d2d8f43ab571.png">

Figure 3: 0-500 is the price range for a large number of items. There are outliers with large value of price. This plot also shows that the items from price range 0-500 had been sold more than expensive items from its height and because each individual id in the train dataset represents sale.


Figure 4:

<img width="775" alt="Screenshot 2022-12-13 163024" src="https://user-images.githubusercontent.com/89792366/207458777-0d5da835-5bb5-4c54-9543-c941ba07e0e2.png">

Figure 4:The sale was the highest in 2013 and the lowest in 2015. Looking at the graphs above the the product sale has been decreasing and the expected count in month in my opinion should be less.


### Problem Formulation

* Data:
  * Input: The input is the dataset with added features explained in Preprocessing/Cleanup
  * Output: item_cnt_month clipped into [0,20] range
  
  * Models
  
    * XGBRegressor: The XGBRegressor generally classifies the order of importance of each feature used for the prediction. A benefit of using gradient boosting is that after the boosted trees are constructed, it is relatively straightforward to retrieve importance scores for each attribute. Also, gradient boosting helps in time-series analysis. As the dataset in this project has time-series variables, I have used this model.
    
    * LSTM: It is used for time-series data processing, prediction, and classification. LSTM leads to many more successful runs, and learns much faster. LSTM also solves complex, articial long time lag tasks that have never been solved by previous recurrent network algorithms. As the dataset in this project has lot of variables to classify with time-series data, I also used LSTM model.
    
    * LightGBM: LightGBM is a gradient boosting framework based on decision trees to increases the efficiency of the model and reduces memory usage. It is based on decision tree algorithms and used for ranking, classification. Thus, to improve time and efficiency with grade boosting, I used this model.

### Training
* Software used:
   * Python packages: numpy, pandas, math, sklearn, seaborn, matplotlib.pyplot, xgboost, lightgbm, joblib, keras
   
* XGB Model:
  The model was created as follows:
  <img width="479" alt="Screenshot 2022-12-13 165801" src="https://user-images.githubusercontent.com/89792366/207463278-9819a6b0-8e6c-431b-8441-e8299077cec6.png">

  The model was trained with fit method:
  <img width="488" alt="Screenshot 2022-12-13 165940" src="https://user-images.githubusercontent.com/89792366/207463384-93749b4a-6199-4cac-a67e-17039d62af37.png">
  
   <img width="444" alt="Screenshot 2022-12-13 170836" src="https://user-images.githubusercontent.com/89792366/207464634-5cb2ec69-e8e1-4f6c-bfd2-88c3e7f32219.png">

  The feature importance plot was plotted with their f-score: 
  <img width="463" alt="Screenshot 2022-12-13 170032" src="https://user-images.githubusercontent.com/89792366/207463504-29f4513b-867e-45e1-bb3e-b830ff6c990b.png">
  date_shop_avg_item_cnt - Mean no. of items sold per month for each month and item_category_id has the     highest f-score. While type code has the lowest f-score

* LightGBM Model:
  The model was created as follows:
  <img width="479" alt="image" src="https://user-images.githubusercontent.com/89792366/207463928-2aa8b325-071f-4045-a784-9442dd82e7b1.png">
  
  The model was trained with fit method:
  <img width="524" alt="Screenshot 2022-12-13 170438" src="https://user-images.githubusercontent.com/89792366/207464117-f5f6e35a-e406-4409-b88c-76dd0c289b9b.png">
  
  <img width="475" alt="Screenshot 2022-12-13 170801" src="https://user-images.githubusercontent.com/89792366/207464523-e771c1bf-8417-4762-8304-d5f46e04c9df.png">

   The feature importance plot was plotted with their f-score:
   <img width="467" alt="Screenshot 2022-12-13 170554" src="https://user-images.githubusercontent.com/89792366/207464278-8222f476-90b7-45f8-ab74-5823935be4e8.png">
   date_shop_cat_avg_item_cnt - Mean no. of items sold per month for each month and shop_id and item_category_id has the highest f-score. While city code has the lowest f-score

* LSTM Model:
   The model was created as follows:
  <img width="763" alt="Screenshot 2022-12-13 171024" src="https://user-images.githubusercontent.com/89792366/207464859-e74a868d-3c63-4643-90f0-b4fce5e3e899.png">
  
  <img width="419" alt="Screenshot 2022-12-13 171101" src="https://user-images.githubusercontent.com/89792366/207464946-e91a35a8-c246-479e-9c75-9b97dd10af12.png">

  The model was trained with fit method:
  <img width="818" alt="Screenshot 2022-12-13 171150" src="https://user-images.githubusercontent.com/89792366/207465073-25c3f2d2-89c7-4c1b-b647-0ca9db8daddb.png">

   The train/valid loss plot:
   
   <img width="302" alt="Screenshot 2022-12-13 171230" src="https://user-images.githubusercontent.com/89792366/207465231-929c79f9-81b2-449a-a37f-c09368f668d6.png">
   
   Validation loss is consistently lower than the training loss, the gap between them remains more or less the same size. It means the model is more accurate on the training model than validation or test.

### Performance Comparison

* The performance metric is root mean squared error(RMSE).
* Table:

![image](https://user-images.githubusercontent.com/89792366/207466372-fc98b29d-d2a3-4cb3-b682-2dbd57d1ac4d.png)
The  following plots show the difference between the difference between true Y_valid and predicted Y_pred from X_valid data.
* XGB plot:
  <img width="490" alt="Screenshot 2022-12-13 172621" src="https://user-images.githubusercontent.com/89792366/207467083-1c39f8f5-5271-4628-959d-ed7a62f4a048.png">
  
* LightGBM plot:
  <img width="481" alt="Screenshot 2022-12-13 172711" src="https://user-images.githubusercontent.com/89792366/207467187-1c45fb53-1680-4547-b978-85539f80e0ea.png">

* LSTM plot:
  <img width="485" alt="Screenshot 2022-12-13 172748" src="https://user-images.githubusercontent.com/89792366/207467250-525d98da-63ee-4723-b9dc-8071b17b3302.png">


### Conclusions

*  From the plots it is seen that the XGBRegressor model was the best among the 3 models as the predicted data was closer to the true value and condensed than the other models. But still, there is large variation from the true value of validation data in XGB model. Overall, the XGB model was not as effective.

### Future Work

* In future, I can make the models more effective interms of getting accurate predictions. This can be done by deeply understanding the features used and advancing parameters used in models. 

### How to reproduce results

* To reproduce the results:
   * import XGBRegressor, lightgbm, and LSTM models.
     The followings commands can be used to import:
     from xgboost import XGBRegressor
     from xgboost import plot_importance
     import lightgbm as lgb
     from keras.models import Sequential
     from keras.layers import LSTM, Dense, Dropout
     from keras.optimizers import Adam
   * Create the train, valid, test dataset as described:
   
   <img width="749" alt="Screenshot 2022-12-13 192023" src="https://user-images.githubusercontent.com/89792366/207481413-75bd2f37-aacc-49c7-afcb-56dea2f3413c.png">

   * Create model as described in Training Section.
   * Train the model as described in Training Section.
   * The predictions can be made as follows for valid and test data to get the feature importance and train/valid loss plot and RMSE.
    Feature Importance Plot:
    <img width="478" alt="Screenshot 2022-12-13 192408" src="https://user-images.githubusercontent.com/89792366/207481935-8d56a610-03f1-4a8a-99a3-3c400cd815a7.png">
    
    Train/Valid Loss Plot:
    
<img width="761" alt="Screenshot 2022-12-13 192455" src="https://user-images.githubusercontent.com/89792366/207482023-0c03c8bf-e600-403c-a4f0-2ffa7fdfc52b.png">

   * Finally, you can get the plots that show the difference between the difference between true Y_valid and predicted Y_pred from X_valid data as follows:
    <img width="471" alt="Screenshot 2022-12-13 192906" src="https://user-images.githubusercontent.com/89792366/207482555-c8fb482a-4254-4360-90a3-06cd1920ccc0.png">

<img width="474" alt="Screenshot 2022-12-13 193008" src="https://user-images.githubusercontent.com/89792366/207482665-13a3c1c6-3ade-44f0-888c-979690dc8b41.png">

    Repeat this method for other models.
    
### Overview of files in repository

  * preprocess.ipynb: Takes input data in CSV and writes out data frame after cleanup.
  * visualization.ipynb: Creates various visualizations of the data.
  * training-model-1.ipynb: Trains the first model and saves model during training.
  * training-model-2.ipynb: Trains the second model and saves model during training.
  * training-model-3.ipynb: Trains the third model and saves model during training.
  * performance.ipynb: loads multiple trained models and compares results.
  * inference.ipynb: loads a trained model and applies it to test data to create kaggle submission.
  * submission_lstm.csv - file contaning item_cnt_month for the test data for lstm model.
  * submission_lgb.csv - file contaning item_cnt_month for the test data for lightgbm model.
  * submission_xgb.csv - file contaning item_cnt_month for the test data for xgbregressor model.

### Software Setup
* Python packages: numpy, pandas, math, sklearn, seaborn, matplotlib.pyplot, xgboost, lightgbm, joblib, keras
* Download seaborn in jupyter - pip install seaborn
* Download lightgbm in jupyter - pip install lightgbm
* Download tensorflow in jupyter - pip install tensorflow
* Download xgboost in jupyter - pip install xgboost

### Data

* Download data files required for the project from the following link:
  https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/data


## Citations

* https://github.com/waylongo/predict-future-sale
* https://www.kaggle.com/code/cocoyachi/lightgbm-futuresales






