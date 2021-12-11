import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd

#IC50 Data from Drug Dataset for Box-Plot Graph:
Erlotinib = [2.91,0.66,-0.14,1.43,1.79,4.19,4.10,3.66,3.69,5.38,3.53,4.28,0.50,3.51,4.11,3.77,3.02,2.15,2.62,5.41,2.58,2.32,-2.16,3.18,2.79,2.10,-1.21,2.52,-0.54,3.54,4.14,1.79,1.41,0.25,2.78,3.16,4.59,2.51,1.81,3.65,-2.74,1.81,3.14,4.26,-1.52,4.16,3.12,0.17,2.11,0.96,1.30]
Gefitinib=[2.53,3.05,2.83,3.76,1.76,2.21,2.89,2.48,2.95,1.88,3.18,2.45,2.44,2.78,2.80,3.47,2.69,0.83,2.75,2.22,2.76,2.98,2.48,2.03]
Alectinib=[2.94,2.28,3.84,2.71,3.09,2.63,2.57,2.58,-0.09,2.50,2.06,1.40,2.11,3.59,2.49,3.27,3.12,2.24,2.63,4.95,-0.53,-1.81,-0.10,2.84,3.77,1.75,1.56,2.01,1.45,0.63,2.13,0.70,1.40,2.87,1.11,0.00,-0.27,2.74,1.82,-0.70,2.60,4.53,4.91,3.17,5.07,5.45,3.33]
Selumetinib=[4.32,3.92,2.98,4.67,1.55,4.46,3.64,0.28,2.55,3.49,3.26,3.47,4.17,4.17]
Crizotinib=[4.01,4.79,5.62,5.51,4.05,0.41,6.64,2.60,1.08,2.83,3.76,3.22,2.42,3.20,4.51,3.31,4.21,3.90,3.93,4.10,3.16,3.44,4.68,3.60,5.61,3.81,5.99,5.61,4.23,5.28,3.04,-0.92,4.57,3.57,4.17,3.93,2.91,4.18,3.35,-0.88,4.73,1.09,4.09,4.60,3.08,3.73,2.87]
Ceritinib=[4.94 ,4.28 ,6.84 ,4.71 ,6.09 ,5.63 ,5.57 ,6.58 ,-0.09 ,4.50 ,2.06 ,1.40 ,6.11 ,3.59 ,7.49 ,4.27 ,6.12 ,4.24 ,5.63 ,5.95 ,0.53 ,1.81 ,3.10 ,2.84 ,6.77 ,1.75 ,5.56 ,6.01 ,2.45 ,1.63 ,6.13 ,2.70 ,3.40 ,5.87 ,3.11 ,3.00 ,-0.27 ,2.74 ,3.82 ,2.70 ,4.60 ,4.53 ,4.91 ,3.17 ,5.07 ,5.45 ,3.33]

#Drug-A & Drug-B Data from the dataset for Drug - Drug Interaction
Data = {
    'Crizotinib': [4.01,4.79,5.62,5.51,4.05,0.41,6.64,2.60,1.08,2.83,3.76,3.22,2.42,3.20,4.51,3.31,4.21,3.90,3.93,4.10,3.16,3.44,4.68,3.60,5.61,3.81,5.99,5.61,4.23,5.28,3.04,-0.92,4.57,3.57,4.17,3.93,2.91,4.18,3.35,-0.88,4.73,1.09,4.09,4.60,3.08,3.73,2.87],
    'Ceritinib': [4.94,4.28,6.84,4.71,6.09,5.63,5.57,6.58,-0.09,4.50,2.06,1.40,6.11,3.59,7.49,4.27,6.12,4.24,5.63,5.95,0.53,1.81,3.10,2.84,6.77,1.75,5.56,6.01,2.45,1.63,6.13,2.70,3.40,5.87,3.11,3.00,-0.27,2.74,3.82,2.70,4.60,4.53,4.91,3.17,5.07,5.45,3.33]
    }

#Dataset read from excel:
drugdata = pd.read_excel (r'C:\Users\Raja Vignesh\Downloads\Final_Drugs_Dataset_V3.xlsx')
#print (drugdata)
df = DataFrame(Data, columns=['Crizotinib', 'Ceritinib'])

finaldrugs = pd.DataFrame(drugdata,columns= ['IC50', 'AUC','RMSE','Z score','Max conc'])
#print (finaldrugs)

X = finaldrugs[['IC50', 'AUC','RMSE','Z score']]
y = finaldrugs['Max conc']

#Spliting data into testing and training:
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

print('==========K-NearestNeighbor=========')

#Initialize and fit the Model:
KNeighbor = KNeighborsClassifier()
y_pred_KNN = KNeighbor.fit(X_train, y_train).predict(X_test)

#Using Confusion matrix to plot the prediction
confusion_matrix = pd.crosstab(y_test, y_pred_KNN, rownames=['Actual'], colnames=['Predicted'])
#Heat_Map of NSCL Cancerdrugs
sn.heatmap(confusion_matrix, annot=True, cmap="YlGnBu")
plt.title("Naiv Bayes Prediction Metrics",fontsize=13.5)
plt.show()

print('K-Nearest Neighbor Results:')
#Accuracy:
print('Accuracy: %f' % metrics.accuracy_score(y_test, y_pred_KNN))
#Precision:
print('Precision: ',metrics.precision_score(y_test, y_pred_KNN, average='macro'))
#Recall:
print('Recall: ', metrics.recall_score(y_test, y_pred_KNN))
#F1-Score:
print('F1-Score: ',metrics.f1_score(y_test, y_pred_KNN, average='micro'))


print('=============Naive_Bayes============')

#Initialize and fit the Model:
Naive_Bayes = GaussianNB()
y_pred_NB = Naive_Bayes.fit(X_train, y_train).predict(X_test)

#Using Confusion matrix to plot the prediction
confusion_matrix = pd.crosstab(y_test, y_pred_NB, rownames=['Actual'], colnames=['Predicted'])
#Heat_Map of NSCL Cancerdrugs
sn.heatmap(confusion_matrix, annot=True, cmap="YlGnBu")
plt.title("Naiv Bayes Prediction Metrics",fontsize=13.5)
plt.show()

print('Naive Bayes Results:')
#Accuracy:
print('Accuracy: %f' % metrics.accuracy_score(y_test, y_pred_NB))
#Precision:
print('Precision: ',metrics.precision_score(y_test, y_pred_NB, average='macro'))
#Recall:
print('Recall: ', metrics.recall_score(y_test, y_pred_NB))
#F1-Score:
print('F1-Score: ',metrics.f1_score(y_test, y_pred_NB, average='micro'))

print('============Random_Forest===========')

#Initialize and fit the Model:
Random_Forest = RandomForestClassifier(n_estimators=100)
Random_Forest.fit(X_train,y_train)
y_pred_RF=Random_Forest.predict(X_test)

#Using Confusion matrix to plot the prediction
confusion_matrix = pd.crosstab(y_test, y_pred_RF, rownames=['Actual'], colnames=['Predicted'])
#Heat_Map of NSCL Cancerdrugs
sn.heatmap(confusion_matrix, annot=True, cmap="PuBuGn")
plt.title("Random_Forest Prediction Matrics",fontsize=13.5)
plt.show()

print('Random Forest Results:')
#Accuracy:
print('Accuracy: %f' % metrics.accuracy_score(y_test, y_pred_RF))
#Precision:
print('Precision: ',metrics.precision_score(y_test, y_pred_RF, average='macro'))
#Recall:
print('Recall: ', metrics.recall_score(y_test, y_pred_RF))
#F1-Score:
print('F1-Score: ',metrics.f1_score(y_test, y_pred_RF, average='micro'))

print('=========Logistic_Regression========')

#Initialize and fit the Model:
logistic_regression= LogisticRegression()
logistic_regression.fit(X_train,y_train)
#Make prediction on the test set:
y_pred_LR=logistic_regression.predict(X_test)

#Using Confusion matrix to plot the prediction
confusion_matrix = pd.crosstab(y_test, y_pred_LR, rownames=['Actual'], colnames=['Predicted'])
#Heat_Map of NSCL Cancerdrugs
sn.heatmap(confusion_matrix, annot=True, cmap="YlOrRd")
plt.title("Logistic_Regression Prediction Matrics",fontsize=13.5)
plt.show()

print('Logistic Regression Results:')
#Accuracy:
print('Accuracy: %f' % metrics.accuracy_score(y_test, y_pred_LR))
#Precision:
print('Precision: ',metrics.precision_score(y_test, y_pred_LR, average='macro'))
#Recall:
print('Recall: ', metrics.recall_score(y_test, y_pred_LR))
#F1-Score:
print('F1-Score: ',metrics.f1_score(y_test, y_pred_LR, average='micro'))

print('==========Box_Plot on Drugs=========')

#Box_Plot of NSCL Drugs:
box_plot_data=[Erlotinib,Gefitinib,Alectinib,Selumetinib,Crizotinib,Ceritinib]
plt.boxplot(box_plot_data,notch=False,patch_artist=True,labels=['Erlotinib','Gefitinib','Alectinib','Selumetinib','Crizotinib','Ceritinib'])
plt.xlabel("Non-Small Cell Lung Cancer Drugs")
plt.ylabel("Measurements in μM")
plt.title("Drug A AND Drug B Interactions",fontsize=13.5)
plt.show()


print('========KMean_Clustering=========')
kmeans = KMeans(n_clusters=4).fit(df)
centroids = kmeans.cluster_centers_
print('Mean Values')
print(centroids)

#Scatter Plot of Drun-A & Drug-B:
plt.scatter(df['Crizotinib'], df['Ceritinib'], c=kmeans.labels_.astype(float), s=100, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200)
plt.xlabel("Drug-A : Crizotinib")
plt.ylabel("Drug-B : Ceritinib")
plt.title("Drug A AND Drug B Interactions",fontsize=13.5)
plt.show()

