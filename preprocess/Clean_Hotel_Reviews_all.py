import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import pickle

def clean_tokens_and_features(df):
    def get_days_stayed(x): 
        nstay_loc=x.find('Stayed')
        if nstay_loc==-1:
            #print(-1)
            return np.nan
        else:
            return int(x[nstay_loc+7:nstay_loc+9])
    df["Negative_Review"]=df["Negative_Review"].apply(lambda x: '' if x=='No Negative' else x)
    df["Review"] = df["Negative_Review"] + df["Positive_Review"]
    df["Days_stayed"]=df["Tags"].apply(get_days_stayed)
    df["Hotel_location"]=df["Hotel_Address"].apply(lambda x:' '.join(x.split()[-2:]))
    df["Review"]=df["Review"].replace({
        "No Negative": np.nan,
        "No Positive": np.nan
    })
    
    df=df.dropna(subset=['Review']) ##some of the reviews are nan 
    #remove stopwords,digits and apply stemming 
    all_cleaned_rev_tokens=[]
    for review in df['Review'].values:
        review=review.strip()
        tokens=nltk.word_tokenize(review.lower()) 
        tokens_clean=[PorterStemmer().stem(tok) for tok in tokens if tok not in stopwords.words('english') and tok.isdigit()==False]
        all_cleaned_rev_tokens.append(tokens_clean)
        
    #transform reviews back to text after cleaning 
    cleaned_reviews_text=[]
    for rev in all_cleaned_rev_tokens:
        cleaned_reviews_text.append(' '.join(rev))
    
    df['Cleaned_Review'] = cleaned_reviews_text
    print(df.shape) 
    df=df[df['Cleaned_Review']!='']
    print(df.shape) 
    df=df.reset_index(drop=True)
    #only get tokens that appear in more than 2 docs 
    tfidf_vectorizer = TfidfVectorizer(min_df=2)
    tfidf_vectorizer.fit_transform(df['Cleaned_Review'].values) #cleaned_reviews_text
    
    word2id={}
    for id,word in enumerate(tfidf_vectorizer.get_feature_names()):
        word2id[word]=id
    
    #after filter by tfidf threshold, how many vocabulary we have?  
    vocabulary_size=len(word2id)
    print('vocabulary size:{}'.format(vocabulary_size))
    
    #transform review to token id 
    cleaned_rev_nums=[]
    for review in df['Cleaned_Review'].values:
        rev_num=[]
        for tok in nltk.word_tokenize(review):
            if tok in word2id:
                rev_num.append(word2id[tok])
            else:
                pass
        cleaned_rev_nums.append(rev_num)
    
    cleaned_rev_length=[len(i) for i in cleaned_rev_nums]
    row_to_remove=[i for i,v in enumerate(cleaned_rev_nums) if len(v)==0]
    new_df=df[df.index.isin(row_to_remove)==False]
    
    #define sentiment class 
    def score_split(x):
        if (x>=0 and x<=4.9):
            return 'neg'
        elif (x>= 5 and x<=7.5):
            return 'ok'
        elif (x> 7.5 ):
            return 'pos'
        
    new_df['Review_Class']=new_df['Reviewer_Score'].apply(lambda x: score_split(x))
    print(new_df['Review_Class'].value_counts())
    return new_df

def downsample_classes(new_df):
    #downsample to balance number of samples for each class
    min_class_cnt=min(new_df['Review_Class'].value_counts())
    pos_sampled=new_df[new_df['Review_Class']=='pos'].sample(min_class_cnt,random_state=123)
    neg_sampled=new_df[new_df['Review_Class']=='ok'].sample(min_class_cnt,random_state=123)
    new_df_sampled=pd.concat([pos_sampled,neg_sampled,new_df[new_df['Review_Class']=='neg']],axis=0)
    print(new_df_sampled['Review_Class'].value_counts())
    new_df_sampled.shape 
    #create review sequence ids and word2id mapping 
    review_tokenize=[nltk.word_tokenize(i) for i in new_df_sampled['Cleaned_Review']]
    all_word=set([word for rev in review_tokenize for word in rev])
    word2id={word:i for i,word in enumerate(all_word)}
    vocabulary_size=len(word2id)
    return new_df_sampled,word2id,review_tokenize


def review_to_tokenid(review_tokenize,word2id,new_df_sampled):
    cleaned_rev_nums=[]
    for review in review_tokenize:
        rev_num=[]
        for tok in review:
            rev_num.append(word2id[tok])
        cleaned_rev_nums.append(rev_num)
    return cleaned_rev_nums
    #cleaned_rev_length=[len(i) for i in cleaned_rev_nums]
        
def save_files(clean_df,cleaned_rev_nums,word2id,new_df_sampled):
    clean_df.to_csv('Clean_Hotel_Reviews_tfidf_remove.csv',index=False)
    new_df_sampled.to_csv("Clean_Hotel_Reviews_tfidf_remove_sampled.csv",index=False)
    #save to pickle
    outfile = open('ordered_review_tokenid_sampled.pkl','wb')
    pickle.dump(cleaned_rev_nums,outfile)
    outfile.close()
    
    outfile = open('word2id_sampled.pkl','wb')
    pickle.dump(word2id,outfile)
    outfile.close()

def main():
    df=pd.read_csv('Hotel_Reviews.csv')
    clean_df=clean_tokens_and_features(df)
    new_df_sampled,word2id,review_tokenize=downsample_classes(clean_df)    
    cleaned_rev_nums=review_to_tokenid(review_tokenize,word2id,new_df_sampled)
    save_files(clean_df,cleaned_rev_nums,word2id,new_df_sampled)

if __name__ == "__main__":
    main()
