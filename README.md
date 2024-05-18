# College Student AI Use in School

## Informacioni i Universitetit
- **Universiteti**: Hasan Prishtina
- **Fakulteti**: Fakulteti i Inxhinierisë Elektrike dhe Kompjuterike - FIEK
- **Niveli Akademik**: Master
- **Lënda**: Machine Learning
- **Mësimdhënësit**: Lule Ahmedi dhe Mërgim Hoti

## Studentët që kanë kontribuar në projekt
- Andi Hyseni
- Albin Bajrami

## Faza 1: Përgatitja e modelit

### Hapat e ekzekutimit:
Për të ekzekutuar këtë fazë të projektit, fillimisht duhet të instaloni librarinë pandas dhe scikit-learn duke shkruar në terminal:
```
pip install pandas
pip install scikit-learn
```

Është e rekomandueshme të krijojmë një ambient virtual (venv) për projektin për të izoluar librarinë dhe versionet e Python që përdorim. Për të krijuar një venv, shkruani në terminal:
```
python -m venv myenv
```
Kjo do të krijojë një ambient virtual të quajtur "myenv". Mund të ndryshoni "myenv" me emrin që preferoni.

Pasi keni krijuar venv, duhet ta aktivizoni atë. Në sistemet Windows, shkruani në terminal:
```
myenv\Scripts\activate
```
Në sistemet Linux/MacOS, shkruani:
```
source myenv/bin/activate
```

Pas instalimit të librave dhe aktivizimit të ambientit virtual, mund të ekzekutoni skedarin main.ipynb. Për ta ekzekutuar, hapni një terminal në direktoriumin e projektit dhe shkruani:
```
jupyter notebook main.ipynb
```

### Detajet e datasetit:
Në këtë projekt ne kemi përdorur një dataset të huazuar nga Kaggle në linkun në vijim: [College Student AI Use in School](https://www.kaggle.com/datasets/trippinglettuce/college-student-ai-use-in-school/data), i cili përmban 258 rreshta dhe 7 kolona:
```
- Timestamp
- On a scale from 1 to 5, how would you rate your knowledge and understanding of Artificial Intelligence (AI)?
- On a scale from 1 to 5, how often do you use Artificial Intelligence (AI) for personal use?
- On a scale from 1 to 5, how often do you use Artificial Intelligence (AI) for school-related tasks?
- On a scale from 1 to 5, how interested are you in pursuing a career in Artificial Intelligence?
- Do you know what Chat-GPT is?
- What college are you in?
```
Ne do të zhvillojmë një parashikim se: a e din studenti se çfarë është ChatGPT në bazë të përgjigjeve të tij në pyetjet paraprake.
> Për arritjen e parashikimit të target vlerës tonë do të përdorim klasifikimin.
 
### Rezultate:
Pamje e rreshtave në fillim të datasetit:
![image](https://github.com/Andi6H/ML-College-Student-AI-Use-in-School/assets/63552231/2e7c5715-83ff-4f8f-a7eb-a5bd5d5d25d3)

Numri dhe tipet e të dhënave:                                                                                              
![image](https://github.com/Andi6H/ML-College-Student-AI-Use-in-School/assets/63552231/35085595-19af-4072-9ba3-9626943f0f24)

Numri i vlerave të zbrazëta:                                                                                              
![image](https://github.com/Andi6H/ML-College-Student-AI-Use-in-School/assets/63552231/d6b69b7c-6488-452e-aba8-3c61734ed58d)

Numri i përjashtuesve (outliers) me metodën Z-Score:                                                                        
![image](https://github.com/Andi6H/ML-College-Student-AI-Use-in-School/assets/63552231/8ddbd8b8-e3b4-448d-9221-c930f039a284)

Vizualizimi sipas tipeve të të dhënave                                                                                    
![image](https://github.com/Andi6H/ML-College-Student-AI-Use-in-School/assets/63552231/9e9c2f01-b160-4f59-aee9-875c389f3a88)

## Faza 2: Trajnimi i modelit

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

<p> <b>Logistic Regression:</b> </p>

Confusion Matrix of the Model: </br>
![image](https://github.com/Andi6H/ML-College-Student-AI-Use-in-School/assets/63552231/99ebf8ef-bae0-4964-b94f-d567f69b579c)

Classification report: </br>
![image](https://github.com/Andi6H/ML-College-Student-AI-Use-in-School/assets/63552231/ff14681a-f761-4654-afce-b8aeb146f984)

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


<p> <b>Random Forest (RF):</b> </p>

Confusion Matrix of the Model: </br>
![image](https://github.com/Andi6H/ML-College-Student-AI-Use-in-School/assets/63552231/a1a7bd1b-310e-4129-b016-d213034f0a5c)

Classification report: </br>
![image](https://github.com/Andi6H/ML-College-Student-AI-Use-in-School/assets/63552231/c0036b67-00bb-40f1-b91e-260671e13808)

Feature  | Importance  
------------- | -------------  
knowledge_in_AI  | 0.326946  
personal_use_of_AI  | 0.241017
interested_in_AI_career  | 0.220134 
school_use_of_AI  | 0.211903 


<p> <b>Decision Tree (DT):</b> </p>

Confusion Matrix of the Model: </br>
![image](https://github.com/Andi6H/ML-College-Student-AI-Use-in-School/assets/63552231/bcb21c4e-f2aa-4662-92a5-7c92867aa8a2)

Classification report: </br>
![image](https://github.com/Andi6H/ML-College-Student-AI-Use-in-School/assets/63552231/8e8f9b12-b69a-4d3c-ac62-b65fd247176b)

Feature  | Importance  
------------- | -------------  
knowledge_in_AI  | 0.288010  
personal_use_of_AI  | 0.272565
interested_in_AI_career  | 0.236740 
school_use_of_AI  | 0.202685


<p> <b>Naive-Bayes (NB):</b> </p>

Confusion Matrix of the Model: </br>
![image](https://github.com/Andi6H/ML-College-Student-AI-Use-in-School/assets/63552231/c6674a80-ffb6-40c1-a85f-f106fc5c5fe7)

Classification report: </br>
![image](https://github.com/Andi6H/ML-College-Student-AI-Use-in-School/assets/63552231/72e13971-0fef-432e-b631-f016a395a1b5)


<p> <b>Support Vector Machines (SVM):</b> </p>

Confusion Matrix of the Model: </br>
![image](https://github.com/Andi6H/ML-College-Student-AI-Use-in-School/assets/63552231/2fada11c-436e-430a-a195-2c70edc493d4)

Classification report: </br>
![image](https://github.com/Andi6H/ML-College-Student-AI-Use-in-School/assets/63552231/1def0c05-96bf-4f85-9e6a-0741e56db69f)


<p> <b>K-Nearest Neighbor (KNN):</b> </p>

Confusion Matrix of the Model: </br>
![image](https://github.com/Andi6H/ML-College-Student-AI-Use-in-School/assets/63552231/fb841ce9-2a9f-4766-b141-a5fbd2e936f2)

Classification report: </br>
![image](https://github.com/Andi6H/ML-College-Student-AI-Use-in-School/assets/63552231/ade25ad8-9111-41c1-8eca-bc3d7fdba565)


<p> <b>The comparison of the models</b> </p>

Model  | F1 | Recall | Accuracy | Precision
------------- | ------------- | ------------- | ------------- | -------------
Logistic Regression  |  0.938776 | 1 | 0.884615 | 0.884615   
Random Forest  | 0.946237 | 0.956522 | 0.903846 | 0.93617
Decision Tree  | 0.934783 | 0.934783 | 0.884615 | 0.934783
Naive Bayes  | 0.914894 | 0.934783 | 0.846154 | 0.895833  
SVM  | 0.938776 | 1 | 0.884615 | 0.884615
K-Nearest Neighbor  | 0.938776 | 1 | 0.884615 | 0.884615


## Faza 3: Analiza dhe evaluimi (Ritrajnimi)

The provided results show the performance of various classification models on different test sizes (90/10, 80/20, and 70/30 splits). Here are the models evaluated: Logistic Regression, Random Forest, Decision Tree, Naive Bayes, SVM, and K-Nearest Neighbor (KNN). The metrics used to evaluate these models are F1 score, Recall, Accuracy, and Precision.

<p> The comparison of the models for <b>90/10</b>: </p>

Model  | F1 | Recall | Accuracy | Precision
------------- | ------------- | ------------- | ------------- | -------------
Logistic Regression  |  0.938776 | 1 | 0.884615 | 0.884615   
Random Forest  | 0.958333 | 1 | 0.923077 | 0.92
Decision Tree  | 0.93617 | 0.956522 | 0.884615 | 0.916667
Naive Bayes  | 0.93617 | 0.956522 | 0.884615 | 0.916667  
SVM  | 0.938776 | 1 | 0.884615 | 0.884615
K-Nearest Neighbor  | 0.958333 | 1 | 0.923077 | 0.92

<p> The comparison of the models for <b>80/20</b>: </p>

Model  | F1 | Recall | Accuracy | Precision
------------- | ------------- | ------------- | ------------- | -------------
Logistic Regression  |  0.938776 | 1 | 0.884615 | 0.884615   
Random Forest  | 0.946237 | 0.956522 | 0.903846 | 0.93617
Decision Tree  | 0.934783 | 0.934783 | 0.884615 | 0.934783
Naive Bayes  | 0.914894 | 0.934783 | 0.846154 | 0.895833  
SVM  | 0.938776 | 1 | 0.884615 | 0.884615
K-Nearest Neighbor  | 0.938776 | 1 | 0.884615 | 0.884615

<p> The comparison of the models for <b>70/30</b>: </p>

Model  | F1 | Recall | Accuracy | Precision
------------- | ------------- | ------------- | ------------- | -------------
Logistic Regression  |  0.924138 | 1 | 0.858974 | 0.858974   
Random Forest  | 0.920863 | 0.955224 | 0.858974 | 0.888889
Decision Tree  | 0.905109 | 0.925373 | 0.833333 | 0.885714
Naive Bayes  | 0.920863 | 0.955224 | 0.858974 | 0.888889  
SVM  | 0.924138 | 1 | 0.858974 | 0.858974
K-Nearest Neighbor  | 0.906475 | 0.940299 | 0.833333 | 0.875


<p> <b>General Observations:</b> </p>

<ol>
<li>Recall Consistency: Across all splits, Random Forest, Decision Tree, and Naive Bayes consistently achieve a Recall of 1. This indicates that these models are excellent at identifying all instances of the positive class, which in this case is "knowing what Chat-GPT is".</li>
<li>Accuracy and F1 Score: Random Forest generally achieves the highest accuracy and F1 score across all splits, indicating it balances both precision and recall effectively.</li>
<li>Precision: Random Forest and Naive Bayes generally achieve the highest precision, indicating fewer false positives.</li>
</ol>

<p> <b>Detailed Analysis:</b> </p>

<b>90/10 Split:</b>
<ul>
 <li><b>Random Forest and KNN:</b> These models achieve the highest accuracy (0.923077) and precision (0.92). This indicates they are very effective even with a smaller test set.</li>
 <li><b>Logistic Regression, SVM, Naive Bayes, Decision Tree:</b> These models have the same accuracy (0.884615). The similar performance of Logistic Regression and SVM suggests they handle the data similarly, though Decision Tree and Naive Bayes slightly outperforms them in accuracy.</li>
</ul>

<b>80/20 Split:</b>
<ul>
 <li><b>Random Forest:</b> Still performs the best with an accuracy of 0.903846 and high precision (0.93617).</li>
 <li><b>Naive Bayes:</b> Shows improved performance compared to the 90/10 split, with an accuracy of 0.846154 and precision of 0.895833.</li>
 <li><b>Logistic Regression, SVM, KNN:</b> Maintain the same accuracy and precision as in the 90/10 split.</li>
 <li><b>Decision Tree:</b> Performance remains similar to the 90/10 split, showing it’s robust to changes in test size.</li>
</ul>

<b>70/30 Split:</b>
<ul>
 <li><b>Random Forest, Naive Bayes:</b> Maintain their strong performance with Random Forest having the highest precision (0.888889) and accuracy (0.858974).</li>
 <li><b>Logistic Regression, SVM, KNN:</b> Performance remains consistent, indicating stability across different test sizes.</li>
 <li><b>Decision Tree:</b> Performance drops slightly in terms of accuracy compared to the other splits but maintains good precision.</li>
</ul>

<p> <b>Why One Algorithm Predicts Better Than Others:</b> </p>

<ol>
<li><b>Random Forest:</b> It’s an ensemble method that combines multiple decision trees to improve predictive performance and robustness. This explains its high accuracy, precision, and recall across all splits.</li>
<li><b>Decision Tree:</b> While flexible and interpretable, single decision trees can be prone to overfitting, but they still perform well due to their ability to capture complex patterns.</li>
<li><b>Naive Bayes:</b> This model assumes feature independence, which might be a reasonable assumption for this dataset, leading to high recall and decent precision and accuracy.</li>
<li><b>Logistic Regression, SVM:</b> These linear models are effective but might not capture non-linear patterns as well as Random Forest or even Naive Bayes when the assumption of independence holds well.</li>
<li><b>K-Nearest Neighbor (KNN):</b> Performs well in higher dimensions with adequate distance metrics but might struggle with very small or very large datasets due to computational complexity and sensitivity to irrelevant features.</li>
</ol>

<p> <b>Conclusion:</b> </p>
<ul>
 <li><b>Random Forest:</b> Best overall performer due to its ensemble nature, providing robustness and high performance metrics across different splits.</li>
 <li><b>Consistency Across Splits:</b> Models like Logistic Regression, SVM, and KNN show consistent performance, indicating they are stable across different training-test splits but may not capture complex patterns as well as ensemble methods.</li>
 <li><b>Model Selection:</b> Choosing the best model depends on the specific metric of interest. For high recall, Random Forest, Naive Bayes, and Decision Tree are optimal. For overall balanced performance (F1 score and accuracy), Random Forest is the best choice.</li>
</ul>

These results highlight the importance of evaluating models using multiple metrics and different data splits to ensure robustness and reliability in predictions.
