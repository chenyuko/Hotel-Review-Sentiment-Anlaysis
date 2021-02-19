# Hotel-Review-Sentiment-Anlaysis

### Goal: 
The Goal of the project is to provide a sentiment analysis model to help business classify reviewer's review into 3 categories (positvie/ neutral/ negative) utilizing the large european hotels' data set from Booking.com. 

### Model: 
- Baseline: Naive Bayes trained with BOW and Naive Bayes with TFIDF 
- GRU with embedding layers trained from reviews 
- GRU with user background information and embedding layers trained from reviews 

### How to run the program? 
 1. Please first download the hotel review data from https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe and named it as Hotel_Reviews.csv
 2. Run Clean_Hotel_Reviews_all.py to preprocess and vectorize reviews' data in text format  
 3. Follow and run the program cells in Hotel_RNN_model_final.ipynb

### Data Source: 
https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe
