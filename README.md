<h1> <b>Chat GPT popularity - Machine Learning 2024</b> </h1>

As part of Machine Learning project we will use an AI On-Campus Research Survey Dataset taken by <a href="https://www.kaggle.com/datasets/trippinglettuce/college-student-ai-use-in-school">Kaggle</a>. </br>
While using the dataset above we are trying to predict if a student knows or uses Chat GPT and what are the factors that have an impact in knowing/using Chat GPT by using Machine Learning Algorithms.


**Project steps that we will take:**

<ul>
  <li>Data Preprocessing:
    <ul>
      <li>The dataset will be preprocessed by removing irrelevant columns, handling missing values, checking for outliers and more;
      </li>
    </ul>
  </li>  
  <li>Model Selection:
    <ul>
      <li>For model selection we will use some supervised algorithms and also deep learning. Supervised models that we will use include: Logistic Regression, Random Forest (RF), Decision Tree, Naive-Bayes, Support Vector Machines (SVMs), K-Nearest Neighbor (KNN)</li>
    </ul>
  </li>
  <li>Model Evaluation:
    <ul>
      <li>We will evaluate the performance of the trained model on the testing data using metrics such as accuracy, precision, recall, and F1-score.</li>
        <li>We will try the 70/30 train_test_split and see if the accuracy gets better.</li>
      <li>There will be a comparison between models in a table form</li>
      <li>For the best performing algorithms we will check for the most import features.</li>
    </ul>
  </li>
</ul> 
<hr>

<h5>Results </h5>
<p> Accuracy for 0.3</p>

Model  | 70/30 
------------- | ------------- 
Logistic Regression  | 0.8589743589743589   
Random Forest  | 0.8461538461538461 
Decision Tree  | 0.8333333333333334
Naive Bayes  | 0.8589743589743589  
SVM  | 0.8589743589743589
K-Nearest Neighbor  | 0.8461538461538461

<p> Train accuracy 0.3</p>

Model  | 70/30 
------------- | -------------  
Logistic Regression  | 0.8555555555555555 
Random Forest  | 0.8944444444444445  
Decision Tree  | 0.8944444444444445
Naive Bayes  | 0.8166666666666667
SVM  | 0.8555555555555555
K-Nearest Neighbor  | 0.8555555555555555

<p> Test accuracy for 0.3</p>

Model  | 70/30 
------------- | ------------- 
Logistic Regression  | 0.8589743589743589
Random Forest  | 0.8461538461538461 
Decision Tree  | 0.8333333333333334
Naive Bayes  | 0.8589743589743589
SVM  | 0.8589743589743589 
K-Nearest Neighbor  | 0.8461538461538461

<p> From the results we can see that:</p>
<ul>
    <li>Logistic Regression, Naive Bayes, and SVM have performed the best</li>
    <li>Decision Tree has performed the worst</li>
    <li>The accuracies in Logistic Regression, Naive Bayes, and SVM is almost the same. This could mean different things like:
        <ul>
            <li>Choosing between '1' and '0' if a student knows/uses Chat is relatively simple. Then, it's possible that all three algorithms are able to learn the decision boundary effectively and achieve similar performance.</li>
            <li>The dataset is well-suited for linear models. Logistic Regression and Naive Bayes are both linear models, while SVM can be linear or non-linear depending on the kernel used</li>
            <li>Feature representation is informative. The features used for prediction contain sufficient information to discriminate between the classes, allowing all three algorithms to perform well.</li>
        </ul>
    </li>
</ul>


<p> Calculating feature importance for the best permorming algorithm </p>

<p> Logistic Regression: </p>

Feature  | Importance  
------------- | -------------  
knowledge_in_AI  | 0.766864  
personal_use_of_AI  | 0.091307 
interested_in_AI_career  | 0.088664 
school_use_of_AI  | 0.013021 

<p>From the results of this algorithm we can see that:</p>
<ul>
    <li>The most important feature or attribute is knowledge_in_AI and this shows that knowledge_in_AI plays the biggest role in knowing the Chat GPT platform.</li>
    <li>personal_use_of_AI and interested_in_AI_career are also somewhat an important feature or attribute to knowing Chat GPT.</li>
    <li>The feature school_use_of_AI does not contribute to predicting one's proficiency with ChatGPT. In other words, the amount of time spent using AI in school does not necessarily correlate with knowledge of AI platforms, including ChatGPT, if one lacks foundational knowledge in AI.</li>
</ul>

<hr>
<h5> Technologies </h5>
<ul> 
    <li> PyCharm </li>
    <li>Languages:
        <ul><li>Jupiter Notebook</li></ul></li>
</ul>
<hr>

<h5>Creators</h5>
<ul>
    <li>Albin Bajrami</li>
    <li>Andi Hyseni</li>
</ul>
<hr>
