# -*- coding: utf-8 -*-
"""Mini Project Python(Group 3 people)

Download Dataset

Data set เป็น ไฟล์ Excel มี 135 rows × 17 columns เป็นข้อมูลโภชนาการของเมนู Subway

https://www.kaggle.com/datasets/davinm/subway-restaurant-nutrition-data

# **Data Cleaning**
"""

from google.colab import files
uploaded = files.upload()

import pandas as pd
import numpy as np
df = pd.read_csv('exported_data.csv') #อ่านไฟล์
df

df = df.rename(columns={'Unnamed: 0':'Product'}) #เปลี่ยนชื่อจาก Unnamed : 0 เป็น Product
df

pd.DataFrame(df['Category'].value_counts(dropna=False)) #ตรวจสอบการกระจายการเตรียมการตามหมวดหมู่

df.info() #แสดงสรุปข้อมูล

# แสดงจำนวน null หรือ missing values
isnull_df = df.isnull().sum()  # 0 แสดงว่าไม่มี missing values ถือว่าเป็น complete values
print(isnull_df)

analysisdemo = df.loc[: , ['Product','Category','Calories']] #บอกถึงหมวดหมู่ของเมนูนั้นและแคลเลอรี่ที่ได้รับ
print(type(analysisdemo))
print(analysisdemo)

df.loc[3, :] #เลือกคอลัมน์ที่ผู้ใช้งานสนใจจึงจะเเสดงผลโภชนาการของเเต่ละเมนู

df.describe() #เช็คค่า Summary ของแต่ละคอลัมน์

"""# **Machine Learning**

Decision Tree (Supervised Learning)
"""

import matplotlib
import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

#อ่านไฟล์ที่ชื่อว่า exported_data.csv
df = pandas.read_csv('exported_data.csv')
df = df.rename(columns={'Unnamed: 0':'Product'}) #เปลี่ยนชื่อจาก Unnamed : 0 เป็น Product

# Column ที่สำหรับใช้ในการทำ Decision Tree
features = ['Sugars (g)', 'Total Fat (g)','Carbohydrates (g)','Sodium (mg)','Protein (g)','Vitamin A % DV','Vitamin C % DV']

x = df[features]
y = df['Calories']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(x,y)

tree.plot_tree(dtree, feature_names=features)
plt.rcParams['figure.figsize'] = (400, 200)

"""Clustering (Unsupervised Learning)"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

val = pd.read_csv('exported_data.csv')
x = val['Calories'].tolist()
y = val['Total Fat (g)'].tolist()

print('Calories {}'.format(x))
print('Total Fat (g) {}'.format(y))

# Clustering Algorithm
data = list(zip(x,y))

# จำนวนการแบ่งกลุ่ม
Cluster_Count = 2

hierarchical_cluster = AgglomerativeClustering(n_clusters=Cluster_Count, affinity='euclidean', linkage='ward')
labels = hierarchical_cluster.fit_predict(data)

plt.scatter(x,y, c=labels)
plt.show()
plt.rcParams['figure.figsize'] = (5, 5)

"""# **Data Visualization**"""

import pandas as pd
import io

io.StringIO(uploaded["exported_data.csv"].decode('utf_8'))
df = pd.read_csv(io.StringIO(uploaded["exported_data.csv"].decode('utf_8'))) #อ่านไฟล์ CSV นำข้อมูลเข้า data แบ่งตามแถวเเละขึ้นบรรทัดใหม่แล้วใส่ลงในตัวแปร data ในรูปแบบของ list
print(df) #แสดงผล

"""### Bar Charts"""

#กราฟแสดง Calories ของเมนูต่างๆใน Subway
import matplotlib.pyplot as plt
df = df.rename(columns={'Unnamed: 0':'Product'}) #เปลี่ยนชื่อจาก Unnamed : 0 เป็น Product
plt.bar(df['Product'], df['Calories'], color ='b') #plot (Line Chart) --> bar (Bar Chart) --> scatter (Scatter Plot)
plt.title("Menu vs Calories") #ชื่อกราฟ
plt.xlabel("Menu") #แกน x menu
plt.ylabel("Calories") #แกน y Calories
plt.rcParams["figure.figsize"] = (20,40) #พล็อตโดยใช้ Matplotlib โดยในที่นี้เรากำหนดขนาด กว้าง,สูง
plt.xticks(rotation=90) #เป็นการหมุนเส้นแกน x ในกราฟแสดงผลข้อมูล เพื่อให้ข้อความบนแกน x ไม่ทับกันและอ่านได้ชัดเจนขึ้น โดยการกำหนดค่า rotation ในหน่วยองศา เช่น rotation=90 จะหมุนข้อความบนแกน x ไปทางขวา 90 องศา
plt.show()

"""### Pie Charts"""

#Pie Charts แสดงจำนวนเมนูของอาหารเเต่ละประเภท
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Set Labels
labels = ['Sandwich', 'Breakfast', 'Extra','Salad', 'Wrap']  #ชื่อของข้อมูล
sections = [31, 4, 13, 18, 20, ] #จำนวนเต็มของข้อมูล
colors = ['royalblue', 'yellow', 'chocolate', 'deeppink', 'lime'] #สีที่แสดงใน graph

#Ploting Pie Chart
plt.pie(sections, labels=labels, colors=colors,
        startangle=90, #มุมเริ่มต้นองศา
        explode = (0.1, 0, 0, 0, 0), #ความชิดของ graph pie
        autopct = '%1.2f%%') #แสดงเป็นเลขทศนิยม 2 และเพิ่มเครื่องหมาย %

plt.rcParams["figure.figsize"] = (5,15) #พล็อตโดยใช้ Matplotlib โดยในที่นี้เรากำหนดขนาด กว้าง,สูง
plt.axis('equal') #คำสั่งในการวาดกราฟแบบ pie chartให้มีขาดเท่ากัน และ ดูสมดุลกันทั้งหมด
plt.title('Menu') #ชื่อกราฟ
plt.show()

"""### Stacked Bar Charts"""

#แสดงข้อมูลเมนูแซนวิสในช่วงเช้าและสารอาหารหลักในตอนเช้า
import pandas as pd #Panel Data --> Data Preparation
import matplotlib.pyplot as plt

data = pd.DataFrame({   #pd --> import pandas

    "Protein (g)":[24, 24, 19, 27], #กำหนดข้อมูล Protein, Carbohydrates, Vitamin A, Total Fat
    "Carbohydrates (g)":[45, 45, 44, 46],
    "Vitamin A % DV":[20, 20, 20, 20],
    "Total Fat (g)" :[20, 16, 15, 18],
    },
    index=["Bacon, Egg & Cheese", "Black Forest Ham, Egg & Cheese", "Egg & Cheese", "Steak, Egg & Cheese"] #ชื่อของข้อมูล
)
print(data)

data.plot(kind="barh", stacked = True) #kind --> stacked = True #กำหนด stacked=True จะทำให้แท่งกราฟสามารถทับกันได้ และแสดงผลรวมของค่าของแต่ละ column ในแต่ละ index
plt.title("Breakfastr")
plt.xlabel("Total Sales")
plt.ylabel("Gadgets")
plt.rcParams["figure.figsize"] = (10,5) #size
plt.xticks(rotation=0) #เป็นการหมุนเส้นแกน x ในกราฟแสดงผลข้อมูล เพื่อให้ข้อความบนแกน x ไม่ทับกันและอ่านได้ชัดเจนขึ้น โดยการกำหนดค่า rotation ในหน่วยองศา
plt.show()

"""### Scatter Plot"""

#กราฟบอกปริมาณโภชนาการที่จะได้รับจากเมนูสลัด
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.DataFrame({   #pd --> import pandas

        'Cholesterol (mg)':[60, 30, 105, 45, 50, 40, 45, 45, 55, 180, 60, 50, 120, 45, 50, 40, 30, 0], #ข้อมูล Cholesterol, Sodium, Carbohydrates, Sugars, Protein
        'Sodium (mg)':[910, 550,1000, 830, 870, 780, 350, 460, 360, 1750, 1000, 830, 2050, 610, 620, 370, 520, 75 ],
        'Carbohydrates (g)':[28, 12, 13, 12, 12, 22, 10, 10, 10, 25, 12, 13, 21, 11, 30, 10, 11, 9],
        'Sugars (g)' :[20, 6, 7, 5, 5, 9, 5, 5, 5, 11, 5, 7, 9, 5, 21, 5, 5, 5,],
        'Protein (g)' :[17, 13, 32, 12, 14, 15, 17, 19, 23, 54, 14, 19, 44, 19, 19, 15, 14, 3],
        'Product' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],  #กำหนดลำดับของเมนู
        },

    index=["BBQ Rib", "Black Forest Ham", "Chicken & Bacon Ranch", "Cold Cut Combo", #ชื่อเมนู salad
           "Italian B.M.T. Â®", "Meatball Marinara", "Oven Roasted Chicken", "Roast Beef",
           "Rotisserie-Style Chicken", "Southwest Chipotle Chicken Club", "Spicy Italian", "Steak & Cheese",
           "Steak Club Salad", "Subway ClubÂ®", "Sweet Onion Chicken Teriyaki", "Tuna",
           "Turkey Breast", "Veggie DeliteÂ®"]
)

df = pd.DataFrame(data)
print(df)

plt.scatter(df['Product'], df['Cholesterol (mg)'], label='Cholesterol (mg)', color = 'r') #กำหนดแกน x,y กำหนดสีและรูปแบบของจุด
plt.scatter(df['Product'], df['Sodium (mg)'], label='Sodium (mg)', color = 'g', marker='^')
plt.scatter(df['Product'], df['Carbohydrates (g)'], label='Carbohydrates (g)', color = 'b', marker='x')
plt.scatter(df['Product'], df['Sugars (g)'], label='Sugars (g)', color = 'k', marker='s')
plt.scatter(df['Product'], df['Protein (g)'], label='Protein (g)', color = 'm', marker='o')

plt.rcParams["figure.figsize"] = (5,5) #ฟิกค่ากราฟ
plt.title("Cholesterol vs Sodium vs Carbohydrates vs Sugars vs Protein ") # ชื่อกราฟ
plt.xlabel("Salad") #ชื่อเเกน x
plt.ylabel("Scores") #ชื่อแกน y
plt.legend() #แสดงสมาชิกแต่ละตัวของกราฟมุมบนขวา
plt.show()

"""กราฟบอกปริมาณโภชนาการในเมนู Salad ทุกเมนูของ subway"""
