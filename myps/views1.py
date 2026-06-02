from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def home(request):
    return render(request,"home.html")

def hpprediction(request):
    if request.method=="GET":    
        return render(request,"hpprediction.html")
    else:

        #to recieve input data
        hsize=request.POST.get("hsize")
        hbedroom=request.POST.get("hbedroom")
        hbathroom=request.POST.get("hbathroom")
        hage=request.POST.get("hage")
        hdistance=request.POST.get("hdistance")
        hfloors=request.POST.get("hfloors")

        data=pd.read_csv("/home/vilekh/Desktop/batches_current@2026/ML_Batch34_24feb@2026/django_content/myps/myps/housing_data.csv")

        x=data[['House Size (sqft)', 'Number of Bedrooms', 'Number of Bathrooms','House Age', 'Distance to City Center (miles)', 'Number of Floors']]
        y=data["Price"]

        x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,test_size=0.3,random_state=42)

        model=LinearRegression()

        model.fit(x_train,y_train)

        y_pred=model.predict(x_test)

        accuracy=r2_score(y_test,y_pred)

        newdata=pd.DataFrame({"House Size (sqft)":hsize,"Number of Bedrooms":hbedroom,"Number of Bathrooms":hbathroom,"House Age":hage, "Distance to City Center (miles)":hdistance, "Number of Floors":hfloors},index=[1,2,3,4,5,6])

        testmodel=model.predict(newdata)
        print(testmodel)

        return render(request,"hpprediction.html",{"price":testmodel[0]})

