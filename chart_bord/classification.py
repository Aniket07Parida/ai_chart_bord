import mysql.connector
import pandas as pd

conn = mysql.connector.connect(
    user='root',
    host='localhost',
    database='customer',
    password='Aniket@132'
)

customer_query = ('select * from customer_fixed')
sales_query = ('select * from sales_fixed')

customer_data = pd.read_sql(customer_query,conn)
sales_data = pd.read_sql(sales_query,conn)

customer_table_clean = customer_data.replace(r'^\s*$', pd.NA, regex=True)
count_the_empty_space = customer_table_clean.isna().sum()

sales_table_clean = sales_data.replace(r'^\s*$', pd.NA, regex=True)
count_the_empty_row = sales_table_clean.isna().sum()

sales_data['Rating'] = pd.to_numeric(sales_data['Rating'], errors='coerce')
fill_null = sales_data['Rating'].fillna(sales_data['Rating'].median())

drop_colums = sales_data.drop(columns=['Review_Text','Coupon_Code'])

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

customer_data['Total_Orders'] = pd.to_numeric(customer_data['Total_Orders'], errors='coerce')

customer_data['Total_Spent'] = pd.to_numeric(customer_data['Total_Spent'], errors='coerce')

customer_data['AverageOrderValue'] = (
    customer_data['Total_Spent'] / customer_data['Total_Orders'].replace(0, pd.NA)
)

features = ['Total_Orders', 'Total_Spent', 'AverageOrderValue']

X = customer_data[features].fillna(customer_data[features].median())

scale = StandardScaler()

x_saled = scale.fit_transform(X)

k_means = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

customer_data['Segment'] = k_means.fit_predict(x_saled)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,r2_score

regression_data = sales_data[['Quantity', 'Unit_Price', 'Coupon_Discount', 'Total_Amount']].copy()

regression_data = regression_data.apply(pd.to_numeric, errors='coerce').dropna()

X = regression_data[['Quantity', 'Unit_Price', 'Coupon_Discount']]

y = regression_data['Total_Amount']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=42)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train,y_train)

regression_prediction = model.predict(X_test)

absolute = mean_absolute_error(y_test,regression_prediction)
r2_scores = r2_score(y_test,regression_prediction)

customer_data['Churn'] = (customer_data['Total_Orders'] == 0).astype(int)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

X = customer_data[['Gender','Age','Email','Phone','City', 'State']].copy()

# Encode categorical columns
from sklearn.preprocessing import LabelEncoder

categorical_cols = ['Gender', 'Email', 'Phone', 'City', 'State']
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

y = customer_data['Churn']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train,y_train)

classifer_predict = model.predict(X_test)