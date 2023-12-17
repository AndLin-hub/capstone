import pandas as pd 
import os 
from skimage.transform import resize 
from skimage.io import imread 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import svm 
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

Categories=['chest x-ray ai generated 100','chest x-ray images 100 real'] 
flat_data_arr=[] #input array 
target_arr=[] #output array 
datadir='./' 
#path which contains all the categories of images 
for i in Categories: 
      
    print(f'loading... category : {i}') 
    path=os.path.join(datadir,i) 
    for img in os.listdir(path): 
        img_array=imread(os.path.join(path,img)) 
        img_resized=resize(img_array,(150,150,3)) 
        flat_data_arr.append(img_resized.flatten()) 
        target_arr.append(Categories.index(i)) 
    print(f'loaded category:{i} successfully') 
flat_data=np.array(flat_data_arr) 
target=np.array(target_arr)

#dataframe 
df=pd.DataFrame(flat_data)  
df['Target']=target 
df.shape



#input data  
x=df.iloc[:,:-1]  
#output data 
y=df.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20, 
                                               random_state=42, 
                                               stratify=y) 



# Defining the parameters grid for GridSearchCV 
param_grid={'C':[0.1,1,10,100], 
            'gamma':[0.0001,0.001,0.1,1], 
            'kernel':['rbf','poly']} 
  
# Creating a support vector classifier 
svc=svm.SVC(probability=True) 
  
# Creating a model using GridSearchCV with the parameters grid 
model=GridSearchCV(svc,param_grid)

model.fit(x_train,y_train)

# Testing the model using the testing data 
y_pred = model.predict(x_test) 
  
logisticRegr = LogisticRegression(solver = 'saga', max_iter = 4000)

logisticRegr.fit(x_train, y_train)

score = logisticRegr.score(x_test, y_test)

print( f"logistic Regression Score: {score}")




# Make predictions on entire test data
y_prediction = logisticRegr.predict(x_train)
#Create a confusion matrix
matrix = metrics.confusion_matrix(y_train, y_prediction)
#Visualize the matrix with Seaborn
#Write title to display accuracy score
all_sample_title = 'Accuracy Score: {0}'.format(score)
#Set figure shape
plt.figure(figsize=(9,9))
#Use heatmap
sns.heatmap(matrix, annot=True, fmt=".3f", linewidths=0.5, square=True, cmap="mako")
#Label the plotl
plt.ylabel('Actual Label', size=12)
plt.xlabel('Predicted Label', size = 12)
plt.title(all_sample_title, size = 16)
#print classification report
print(classification_report(y_train, y_prediction))



# Calculating the accuracy of the model 
accuracy = accuracy_score(y_pred, y_test) 
  
# Print the accuracy of the model 
print(f"The model is SVM {accuracy*100}% accurate")

print(classification_report(y_test, y_pred, target_names=['x-ray', 'ai-generated']))


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
accuracyKNN = accuracy_score(y_test, y_pred)
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)
print("KNN-Accuracy:", accuracyKNN)


k_values = [i for i in range (1,31)]
scores = []

scaler = StandardScaler()
X = scaler.fit_transform(x)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X, y, cv=5)
    scores.append(np.mean(score))

sns.lineplot(x = k_values, y = scores, marker = 'o')
plt.xlabel("K Values")
plt.ylabel("Accuracy Score")







while True:
    print("enter parent directory")
    parent = input()
    print("enter image")
    image = input()
    path='./' + parent + '/' + image
    img=imread(path) 
    plt.imshow(img) 
    plt.show() 
    img_resize=resize(img,(150,150,3)) 
    l=[img_resize.flatten()] 
    probability=model.predict_proba(l) 
    for ind,val in enumerate(Categories): 
        print(f'{val} = {probability[0][ind]*100}%') 
    print("The predicted image is : "+Categories[model.predict(l)[0]])          