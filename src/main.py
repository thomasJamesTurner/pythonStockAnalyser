import seaborn as sns
import hashlib
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sqlalchemy import create_engine
import os
import math
from sqlalchemy import text



def standardDeviation(df):
    variance = df.var()
    return math.sqrt(variance)

def dropOutliers(df,nominal,threshold):
    indices_to_drop = []
    
    for i, row in df.iterrows():
        if row['prevDayChangePercent'] > nominal + threshold or row['prevDayChangePercent'] < nominal - threshold:
            print(row)
            indices_to_drop.append(i)
    
    df.drop(indices_to_drop, inplace=True)
    
    return df

def DBconnect(filename):
    dirname = os.path.dirname(__file__)
    os.chdir('../_sensitive_')
    with open(filename, "r") as f:
        details = f.read().split()
    os.chdir(dirname)

    # Construct database URL for SQLAlchemy
    db_url = f"mysql+mysqlconnector://{details[1]}:{details[3]}@{details[5]}/stocks"
    engine = create_engine(db_url)
    return engine

def getQuery(filename):
    return (open(("sql/" + filename), "r")).read()



def is_valid_ticker(db, ticker):
    """Check if the ticker exists in the database using SQLAlchemy."""
    query = text("SELECT COUNT(*) FROM stockData WHERE ticker = :ticker")
    
    with db.connect() as conn:  # Establish a connection
        result = conn.execute(query, {"ticker": getID(ticker, 10000000)}).scalar()
    
    return result > 0  # Returns True if ticker exists

def getID(value, mod):
    hashObj = hashlib.sha256()
    hashObj.update(value.encode('utf-8'))
    hexHash= hashObj.hexdigest()
    ID = int(hexHash,16) % mod
    return ID


def corrMat():
    db = DBconnect("pass.txt")
    sql = "SELECT * FROM company;"
    df = pd.read_sql_query(sql, db)
    db.close()

    df = df.drop(columns=['index','dateCollected'], errors='ignore')
    print(df['income'])
    matrix = df.select_dtypes(include=[float, int]).corr()

    plt.figure(figsize=(10, 8))
    plt.xlabel('X-Axis Label', fontsize=11)  # X-axis label font size
    plt.ylabel('Y-Axis Label', fontsize=11)  # Y-axis label font size
    plt.xticks(fontsize=10)  # X-axis tick label font size
    plt.yticks(fontsize=10)  # Y-axis tick label font size
    
    sns.heatmap(matrix, cmap="Greens", annot=True,annot_kws={"size": 5})

    # Show the plot
    plt.show()

import pandas as pd
import plotly.graph_objects as go
import mysql.connector  # Assuming you're using MySQL



def makeCandleStick():
    db = DBconnect("pass.txt") 

    while True:
        ticker = input("Enter a stock ticker: ").strip().upper()
        if is_valid_ticker(db, ticker):
            break

        print("Invalid ticker. Please try again.")

    
    param = (getID(ticker, 10000000),)
    query = "SELECT * FROM stockData WHERE ticker = %s"
    df = pd.read_sql_query(query, db, params=param)

    fig = go.Figure(data=[go.Candlestick(x=df['dateCollected'],
                                         open=df['priceOpen'],
                                         close=df['priceClose'],
                                         high=df['priceHigh'],
                                         low=df['priceLow'])])
    fig.show()


def makeXGBRModel(estimatiors = 40):
    
    db = DBconnect("pass.txt")
    sql = getQuery("getAllTables.txt")
    df = pd.read_sql_query(sql=sql, con=db)
    df = df.drop(columns=['companyName','companyID','exchangeID', 'index', 'dividendExDate', 'ticker', 'industry','dateCollected'])
    df = df.apply(pd.to_numeric, errors='coerce')
    print(df.head())
    dropOutliers(df,df['prevDayChangePercent'].mean(),standardDeviation(df['prevDayChangePercent']))
    x = df.drop(columns=['prevDayChange','prevDayChangePercent'])
    y = df['prevDayChangePercent']
    
    model = xgb.XGBRegressor(n_estimators = estimatiors)
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
    
    try:
        model.fit(x_train,y_train)
    except Exception as e:
        print('error: ',e)
        return
    y_pred = model.predict(x_test)



    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    

    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

    

    df_residuals = pd.DataFrame({
        'Predicted Values': y_pred,
        'R2': r2,
        'True Values':y_test
    })

    # Create a scatter plot with actual and predicted values
    fig = go.Figure()

    # Add scatter for predicted values
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df_residuals['Predicted Values'],
        mode='markers',
        name='Predicted Values',
        marker=dict(color='blue', size=7)
    ))

    # Add scatter for actual values
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df_residuals['True Values'],
        mode='markers',
        name='Actual Values',
        marker=dict(color='red', size=7)
    ))

    # Add a line connecting predicted and actual values
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df_residuals['Predicted Values'],
        mode='lines+markers',
        name='Line Between Predicted & Actual',
        line=dict(color='green', width=2),
        marker=dict(color='green', size=5)
    ))

    # Add annotation for R² score
    fig.add_annotation(
        x=0.5,  # Position for annotation in x-axis
        y=max(df_residuals['Predicted Values']),  # Position for annotation in y-axis
        text=f'R²: {r2:.2f}',
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-40,
        font=dict(size=14, color="black"),
        bgcolor="white"
    )

    # Update layout for titles and labels
    fig.update_layout(
        title="Predicted vs Actual Values with Line",
        xaxis_title="Index",
        yaxis_title="Values",
        showlegend=True
    )

    # Show the plot
    fig.show()

while True:
    makeCandleStick()