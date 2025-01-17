import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st


data= pd.read_csv("E:\coding\practice\prep\spamDetection\spam.csv")
# print(data.shape)
data.drop_duplicates(inplace=True)
# print(data.shape)

data['Category']= data['Category'].replace(['ham', 'spam'], ['Not Spam','Spam'])
# print(data.head())

mess=data['Message']
cat=data['Category']

(mess_train, mess_test, cat_train, cat_test)= train_test_split(mess, cat, test_size=0.2)

cv=CountVectorizer(stop_words='english')
features=cv.fit_transform(mess_train)

#creating model
model=MultinomialNB()
model.fit(features, cat_train)

#trsting model
features_test=cv.transform(mess_test)
# print(model.score(features_test, cat_test))

# predicting data
def pridict(message):
    input_message=cv.transform([message]).toarray()
    result=model.predict(input_message)
    return result


st.header("Spam Detection")
input_string=st.text_input("Enter Your Message Here")

if st.button('Validate'):
    output=pridict(input_string)
    st.markdown(output)




# output=pridict("Congratulations! You have been selected as the lucky winner of a $1,000 gift card from Amazon. To claim your prize, simply click the link below and complete the quick survey. Act fast, this offer is only available for the next 24 hours!")

