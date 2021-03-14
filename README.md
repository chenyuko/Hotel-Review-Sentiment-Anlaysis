# Hotel-Review-Sentiment-Prediction

### Goal: 
The Goal of the project is to provide a sentiment analysis model to help business classify reviewer's reviews into 3 categories (positvie/ neutral/ negative) utilizing the large European hotels' data set from Booking.com. 

### Model: 
- Baseline: Naive Bayes trained with BOW and Naive Bayes with TFIDF 
- GRU with embedding layers trained from reviews 
- GRU with user background information and embedding layers trained from reviews 

<img src="https://github.com/chenyuko/Hotel-Review-Sentiment-Anlaysis/blob/main/model/GRU.png" alt="drawing" width="150"/> <img src="https://github.com/chenyuko/Hotel-Review-Sentiment-Anlaysis/blob/main/model/GRU_combine.png" alt="drawing" width="150"/>

### How to run the program? 
 1. Please first download the hotel review data from https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe and named it as Hotel_Reviews.csv
 2. Run Clean_Hotel_Reviews_all.py to preprocess and vectorize reviews' data in text format  
 3. Follow and run the steps in Hotel_RNN_model_final.ipynb


### Data: 
- There are 515,000 reviews of European Luxury 5 star hotels in total.
- Source:  https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe

