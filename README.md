# NLP Project
## Sentiment Analysis of tweets
Dataset link is: https://www.kaggle.com/kazanova/sentiment140/download

Get twitter API credentials using the following steps:
1. Create a twitter account.
2. Go to https://apps.twitter.com/and log in with your Twitter user account.
3. This step gives you a Twitter dev account under the same name as your user account. 
4. Click Create New App.
5. Fill out the form, agree to the terms, and click Create your Twitter application.
6. In the next page, click on Keys and Access Tokens tab, and copy your API key and API secret.
7. Scroll down and click Create my access token, and copy your Access token and Access token secret.
8. put the keys in twitter_credentials.py file.
<br>
<b> Dataset contains 80,000 positive and negative data <b>
  
![Figure_1](https://user-images.githubusercontent.com/57863681/163769925-3f7458cd-5968-4c21-b79d-d57e34500f6e.png)

<b> Most common tokens used in the dataset <b>

![Figure_2](https://user-images.githubusercontent.com/57863681/163770031-db320182-a99e-4ff1-8768-247775339bf1.png)

<b> Text processing and tokenizing

![text processing 1](https://user-images.githubusercontent.com/57863681/163770655-73ac2784-fe0e-4b2b-af6d-d82a655cea07.png)
![text processing](https://user-images.githubusercontent.com/57863681/163770665-5df29a72-96ef-4957-b852-70f8ef18ebcb.png)

Three different models are trained using different algorithms and their accuracy is compared <br><br>
<b> Linear SVC algorithm training and confusion matrix </b>

![linear svc](https://user-images.githubusercontent.com/57863681/163770786-4099f34c-b4ab-429e-b419-551a3193c2f2.png)
![Figure_Linearsvc](https://user-images.githubusercontent.com/57863681/163770827-d61f081b-e4f4-4403-ba57-acd6019c0886.png)
  
<b> Logistic Regression training and confusion matrix<b>
  
![LR](https://user-images.githubusercontent.com/57863681/163771156-302939cd-f8eb-4747-ad22-d6177d15e1d7.png)
![Figure_3](https://user-images.githubusercontent.com/57863681/163771184-8c45c9c3-5573-4f1d-b8c3-98c8320badab.png)

<b> Bernaulli Naive Bayes training and confusion matrix<b>

![bernaulli](https://user-images.githubusercontent.com/57863681/163771248-7018413b-2718-4974-8ea3-85468371d997.png)
![Figure_bernaulli](https://user-images.githubusercontent.com/57863681/163771262-8bc193fb-5a77-410e-8d76-ae6e72a3aa40.png)


