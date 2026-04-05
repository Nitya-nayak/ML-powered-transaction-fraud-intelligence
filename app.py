import uuid
import os
import streamlit as st
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import plotly.express as px
import datetime as datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine,text
from supabase import create_client
from sqlalchemy.exc import SQLAlchemyError
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
BUCKET_NAME = os.getenv("BUCKET_NAME")

if not all([SUPABASE_KEY,SUPABASE_URL,DATABASE_URL,BUCKET_NAME]):
    st.error("environment variable not found please check out!")
    st.stop()
    
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping = True,      
    pool_recycle  = 300,       
    connect_args  = {"connect_timeout": 10}
)

supabase = create_client(SUPABASE_URL,SUPABASE_KEY)

def get_or_create_id(email):

    query = "SELECT id FROM users WHERE email =:email"
    with engine.connect() as conn:
        id = conn.execute(text(query),{"email":email}).fetchone()

        if id:
            return id[0]

        query = "INSERT INTO users(username,email,created_at) VALUES(:username,:email,:created_at) RETURNING id"
        result = conn.execute(text(query),{
            "username" : email.split('@')[0],
            "email" : email,
            "created_at" : datetime.datetime.utcnow()
        }).fetchone()
        
        conn.commit()
        return result[0]
    
def validate_file(uploaded_file):
    exten = uploaded_file.name.split(".")[-1]
    df = None
    try:
        if exten == "csv":
            df = pd.read_csv(uploaded_file)
        elif exten == "xlsx":
            df = pd.read_excel(uploaded_file)
        elif exten == "json":
            df = pd.read_json(uploaded_file)
        else:
            return None,"Unsuppoted File!"
    
    except Exception as e:
        st.error(f"Something Going Wrong!-> {e}")

    return df,None

def upload_to_storage(user_id,uploaded_file):
    unique = uuid.uuid4()
    file_extension = uploaded_file.name.split('.')[-1]
    new_filename = f"{unique}.{file_extension}"
    file_path = f"{user_id}_{new_filename}"

    try:
        supabase.storage.from_("files").upload(
            path = file_path,
            file = uploaded_file.getvalue()
        )
    
    except Exception as e:
        st.warning(f"Storage upload failed: {e}")

    public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(file_path)

    return public_url, new_filename

def save_metadata(user_id,file_name,file_path,file_size,file_type):

    query = """INSERT INTO datasets(user_id,file_name,file_path,file_size,file_type)
               VALUES(:user_id,:file_name,:file_path,:file_size,:file_type)
               RETURNING id
        """
    with engine.connect() as conn:
        result =  conn.execute(text(query),{
            "user_id" : user_id,
            "file_name" : file_name,
            "file_path" : file_path,
            "file_size" : file_size,
            "file_type" : file_type
        }).fetchone()

        conn.commit()
        return result[0]
    
def dataset_profiling_Engine(df):
    row_count,column_count = df.shape
    total_missing = df.isnull().sum().sum()
    total_duplicate = df.duplicated().sum()
    memory_usage = df.memory_usage(deep = True).sum()

    return {
        'row_count':int(row_count),
        'column_count':int(column_count),
        'total_missing':int(total_missing),
        'total_duplicate':int(total_duplicate),
        'memory_usage':int(memory_usage)
    }

AMOUNT_COLUMNS = [
"amount","transactionamount","transaction_amount","txn_amount",
"txnvalue","transactionvalue","payment","price","cost","charge",
"bill","purchase_amount","transfer_amount","withdraw_amount",
"deposit_amount","payment_amount","order_amount","paid_amount",
"received_amount","fund","remittance"
]

TIME_COLUMNS = [
"timestamp","datetime","transactiondate","transaction_date",
"event_time","created_at","updated_at","transaction_time","date"
]

BALANCE_COLUMNS = [
"balance","accountbalance","account_balance","available_balance",
"wallet_balance","current_balance","remaining_balance","bank_balance"
]


LOCATION_COLUMNS = [
"location","city","country","state","region","area","place",
"branch","address","geo","geolocation"
]

DEVICE_COLUMNS = [
"device","deviceid","device_id","device_type","mobile",
"phone","terminal","machine","atm","pos","hardware_id"
]

NETWORK_COLUMNS = [
"ip","ipaddress","ip_address","network","host","server",
"connection","gateway"
]

TRANSACTION_TYPE_COLUMNS = [
"type","transactiontype","transaction_type","txn_type",
"payment_type","transfer_type","method","mode","channel"
]

CUSTOMER_COLUMNS = [
"accountid","account_id","customerid","customer_id",
"userid","user_id","clientid","client_id"
]

BEHAVIOUR_COLUMNS = [
"loginattempts","login_attempts","failed_login","attempts",
"transactionduration","transaction_duration","session_time"
]

TEXT_COLUMNS = [
"description","remarks","comment","note","details",
"transaction_note","narration","message","explanation","reason"
]

FRAUD_COLUMNS = [
"fraud","isfraud","is_fraud","fraudlabel","fraud_flag",
"target","class","label","scam","fraud_status"
]

def user_column_detector(df,col,threshold = 89):
    detected = []
    for columns in df.columns:
        cleaned_df_col = columns.lower().replace(" ","").replace("_","")

        for c in col:
            cleanded_system_col = c.lower().replace("_","")

            if cleanded_system_col == cleaned_df_col:
                detected.append(columns)
                break
            
            score = fuzz.ratio(cleanded_system_col,cleaned_df_col)
            if score >= threshold:
                detected.append(columns)
                break
    
    return list(set(detected))
                        

def schema_detected(df):
    schema={}

    schema["amount"] = user_column_detector(df,AMOUNT_COLUMNS)
    schema["balance"] = user_column_detector(df,BALANCE_COLUMNS)
    schema["time"] =  user_column_detector(df,TIME_COLUMNS)
    schema["location"] = user_column_detector(df,LOCATION_COLUMNS)
    schema["devices"] = user_column_detector(df,DEVICE_COLUMNS)
    schema["network"] = user_column_detector(df,NETWORK_COLUMNS)
    schema["transaction"] = user_column_detector(df,TRANSACTION_TYPE_COLUMNS)
    schema["customer"] = user_column_detector(df,CUSTOMER_COLUMNS)
    schema["behaviour"] = user_column_detector(df,BEHAVIOUR_COLUMNS)
    schema["text"] = user_column_detector(df,TEXT_COLUMNS)
    schema["fraud"] = user_column_detector(df,FRAUD_COLUMNS)

    return schema

def fraud_detect_col(df,schema):
    if schema['fraud']:    
        fraud_df = schema["fraud"][0]
        columns = df[fraud_df].value_counts()

        if len(columns) < 2:
            st.warning("Fruad column doesn't contains proper data!")
            st.stop()

        normal_val = columns.index[0]
        fraud_val = columns.index[-1]
    
    else:
        st.warning("fraud column was not detected!")
        return None,None

    return normal_val,fraud_val

def visualization(df,schema,normal_val,fraud_val):
    st.header("🔍 Transaction Fraud Analysis")
    st.caption("Automatically analyzed based on your dataset")

    #Graph no 1 Normal vs Fraud
    if schema["fraud"]:
        st.subheader("📊 How Much Fraud Is There?")
        st.caption("Distribution of fraud vs normal transactions")

        named_col = schema['fraud'][0]

        pie_data = df[named_col].value_counts().reset_index()

        pie_data.columns = ["Type","Count"]

        pie_data["Type"] = pie_data["Type"].map({normal_val:"Normal",fraud_val:"Fraud"})

        figg1 = px.pie(
            pie_data,
            names = "Type",
            values = "Count",
            color_discrete_map = {"Fraud":"red","Normal":"green"}
        )

        st.plotly_chart(figg1)
    else:
        st.warning("Fraud column was not detected!")
    
    #Graph no 2 amount distribution
    if schema["amount"]:
        st.subheader("💰 Do Fraudsters Spend More?")
        st.caption("Comparing transaction amounts — fraud vs normal")

        amount_retrive = schema["amount"][0]

        pie_amount = df[amount_retrive]

        figg2 = px.histogram(
            df,
            x = amount_retrive,
            color = named_col,
            color_discrete_map = {
                fraud_val:"red",
                normal_val:"green"
            }
        )
        st.plotly_chart(figg2)

    else:
        st.warning("Amount Column was not detected!")

    #Graph no 3 top 10 fraud cities

    if schema["location"]:
        st.subheader("📍 Where Is Fraud Happening?")
        st.caption("Top 10 cities with highest fraud activity")

        location_col = schema['location'][0]
        fraud_ = schema['fraud'][0]

        df_fra = df[df[fraud_] == fraud_val]

        loc_df = df_fra[location_col].value_counts().nlargest(10).reset_index()

        loc_df.columns = ["location","count"]

        figg3 = px.bar(
            loc_df,
            x = "count",
            y = "location",
            color = "count",
            color_continuous_scale = "Reds",
            title = "Top 10 Fraud Locations"
        )
        st.plotly_chart(figg3)

    else:
        st.warning("Location columns was not detected!")


    if schema["time"]:
        st.subheader("🕐 When Does Fraud Happen?")
        st.caption("Fraud activity across different hours of the day")

        t = schema['time'][0]
        fra = schema['fraud'][0]

        df['hour'] = pd.to_datetime(df[t], errors= "coerce").dt.hour

        only_fraud = df[df[fra] == fraud_val]

        count_hour = only_fraud['hour'].value_counts().reset_index()

        count_hour = count_hour.sort_values('hour')

        count_hour.columns = ['hour','count']

        figg4 = px.line(
            count_hour,
            x = "hour",
            y = "count",
            markers = True,
            color_discrete_sequence = ["red"],
            title = "Fraud by Hour of Day"
        )
        st.plotly_chart(figg4)
    else:
        st.warning("Time column was not detected!(Please! Cheak dataset)")

    if schema['transaction']:
        st.subheader("🏧 Which Channel Is Most Risky?")
        st.caption("Fraud count across ATM, Online, and Branch")

        fra_col = schema['fraud'][0]
        tran_col = schema['transaction'][0]

        df_new = df[df[fra_col] == fraud_val]

        new_tran = df_new[tran_col].value_counts().reset_index()

        new_tran.columns = ['transaction','count']

        figg5 = px.bar(
            new_tran,
            x = "transaction",
            y = "count",
            color = "transaction",
            title = "Fraud by Channel(ATM/Online/Branch/etc....)"
        )

        st.plotly_chart(figg5)
    
    else:
        st.warning("Transaction (Channel) was not detected! (Please! Check dataset)")
def feature_engineering(df, schema, fraud_val):

    time_col      = schema["time"][0]
    amo_col       = schema["amount"][0]
    balan_col     = schema["balance"][0]
    behaviour_col = schema["behaviour"][0]

    
    df["days"] = pd.to_datetime(df[time_col]).dt.dayofweek
    df["hour"] = pd.to_datetime(df[time_col]).dt.hour 

    df["is_weekend"] = df["days"].isin([5,6]).astype(int)
    df["is_night"]   = df["hour"].between(0,5).astype(int) 

    threshold = df[amo_col].quantile(0.95)
    df["is_high_amount"] = (
        df[amo_col] > threshold
    ).astype(int)

    df["amount_to_balance"] = (
        df[amo_col] / (df[balan_col] + 1)
    )

    df["is_suspicious_login"] = (
        df[behaviour_col] > 3
    ).astype(int)

    if schema["transaction"]:
        for col in schema["transaction"]:
            unique_vals = df[col].str.lower().unique()

            # Credit/Debit
            if any(v in ["credit","debit"]
                   for v in unique_vals):
                df["is_debit"] = (
                    df[col].str.lower() == "debit"
                ).astype(int)

            # Online/ATM/Branch
            if any(v in ["online","atm","branch"]
                   for v in unique_vals):
                df["is_online"] = (
                    df[col].str.lower() == "online"
                ).astype(int)

    if schema["location"] and schema["fraud"]:
        loc_col   = schema["location"][0]
        fraud_col = schema["fraud"][0]
        high_risk = (
            df[df[fraud_col] == fraud_val][loc_col]
            .value_counts().head(10).index.tolist()
        )
        df["is_high_risk_location"] = (
            df[loc_col].isin(high_risk)
        ).astype(int)

    return df

def preprocessing(df,schema):
    fraud_col = schema['fraud'][0]

    selection_col = df.select_dtypes(include="object").columns.astype(str).tolist()

    for i in selection_col:
        if fraud_col in i:
            selection_col.remove(i)

    label_col = []
    ohe_col = []
    skip_col = []

    for col in selection_col:
        count = df[col].nunique()
        
        if count == 2:
            label_col.append(col)

        elif count >= 3 and count <= 5:
            ohe_col.append(col)

        else:
            skip_col.append(col)

    if label_col:

        le = LabelEncoder()

        for c in label_col:
            df[c] = le.fit_transform(df[c])
    
    else:
        st.warning("Categorical data is missing for LabelEncoding!")
    
    if ohe_col:

        ohe = OneHotEncoder(drop = "first",sparse_output = False, handle_unknown = 'ignore')

        df_new = ohe.fit_transform(df[ohe_col])

        names = ohe.get_feature_names_out(ohe_col)

        df_update = pd.DataFrame(
            df_new,
            columns = names,
            index = df.index
        )

        df = df.drop(ohe_col,axis = 1)
        df = pd.concat([df,df_update] , axis = 1)
    
    else:
        st.warning("Categorical data is missing for OneHotEncoding!")

    return df

def model(df,schema,fraud_val):

    fraud_col = schema['fraud'][0]

    X = df.select_dtypes(include = ["float32","float64","int32","int64","uint8"])
    X = X.drop(fraud_col,axis = 1, errors = "ignore")
    X = X.fillna(0)

    Y = df[fraud_col]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,Y,
        test_size = 0.2,
        random_state = 42
    )

    ml = XGBClassifier(
        n_estimators = 100,
        random_state = 42,
        eval_metric = "logloss"
    )

    ml.fit(X_train,Y_train)

    return ml, X , X_test, Y_test

def insights(ml,X,X_test,Y_test,schema,df,fraud_val):

    prob = ml.predict_proba(X)

    try:
        if prob.shape[1] > 1:

            df["fraud_probability"] = prob[:,1]
            df["risk_score"] = (df['fraud_probability'] * 100).round(2)

        else:
            df['fraud_probability'] = prob[:,0]
            df["risk_score"] = (df['fraud_probability'] * 100).round(2)
    
    except Exception as e:
        st.warning(f"Prediction Error -> {e}")

    iso = IsolationForest(
        contamination = 0.05,
        random_state= 42
    )

    df['anomaly'] = iso.fit_predict(X)
    df['anomaly_flag'] = df["anomaly"].map({1:"Normal",-1:"Suspicious"})

    return df

def insight_behavioral_baseline(df,schema):

    try:

        amount_col = schema['amount'][0]
        account_col = schema['customer'][0]

        stats = df.groupby(account_col)[amount_col].agg(
            mean_spend = 'mean',
            std_spend  = 'std'
        ).reset_index()

        df = df.merge(stats,on = account_col, how = "left")

        df['std_spend'] = df['std_spend'].fillna(0)

        df['z_score'] = ((df[amount_col] - df["mean_spend"]) / (df['std_spend'] + 1))

        outlier = df[df['z_score'] > 3]

        col1, col2 = st.columns(2)
        col1.metric("Suspicious Spikes",f"{len(outlier):,}")
        col2.metric("Accounts Affected",f"{outlier[account_col].nunique():,}")

        if not outlier.empty:

            top = outlier.groupby(account_col)['z_score'].max().nlargest(10).sort_values(ascending = True).reset_index()

            figg = px.bar(
                top,
                x = 'z_score',
                y = account_col,
                color = 'z_score',
                color_continuous_scale = 'Reds',
                title = "Top 10 Fraudster Account(Spends more than average amount of Money)"
            )

            st.plotly_chart(figg)
        
        else:
            st.info("✅ No any Suspicious Transactions")

    except Exception as e:
        st.warning(f"Behavioral insight error: {e}")

    return df

def insight2(df,schema):
    try:
        st.subheader("⚡ Transaction Velocity Anomaly")
        st.caption("Transactions happen in same account within 60 sec == Red Flag!")

        account_col = schema["customer"][0] if schema["customer"] else None
        time_col    = schema['time'][0] if schema["time"] else None

        df['ts_'] = pd.to_datetime(df[time_col], errors='coerce')

        df = df.sort_values([account_col,'ts_'])

        df["sec_gap"] = df.groupby(account_col)['ts_'].diff().dt.total_seconds()

        gap_sec = df[(df['sec_gap'] >= 0 ) & (df['sec_gap'] < 60 )]

        bins = pd.cut(
            gap_sec['sec_gap'],
            bins = [0,5,15,30,60],
            labels = ["0-5s","5-15s","15-30s","30-60s"]
        )
        

        bins_df = bins.value_counts().sort_index().reset_index()
        bins_df.columns = ["Bins_range","Counts"]

        figg = px.bar(
            bins_df,
            x = 'Counts',
            y = "Bins_range",
            color = "Counts",
            color_continuous_scale = "Reds",
            title = "Suspicious Transaction within minimal seconds"
        )

        st.plotly_chart(figg)

        with st.expander("🔍 Rapid Transactions :"):
            st.dataframe(
                gap_sec[[account_col,time_col,"sec_gap"]]
                .sort_values("sec_gap").head(20).reset_index(drop = True),use_container_width = True
            )
    except Exception as e:
        st.warning(f"Velocity insight error: {e}")
    
    return df

st.title("ML-powered transaction fraud intelligence System")

email  = st.text_input("Enter your email")
df = None
dataset_id = None

if email:
    uploaded_file = st.file_uploader(
        "Upload dataset",
        type = ['csv','xlsx','json']
    )


    if uploaded_file:

        if st.button("upload"):
            try:
                user_id = get_or_create_id(email)

                df,error = validate_file(uploaded_file)

                if error:
                    st.error(error)
                    st.stop()

                st.subheader("Data preview")
                st.dataframe(df.head())

                public_url,new_filename = upload_to_storage(user_id,uploaded_file)

                dataset_id = save_metadata(
                                user_id = user_id,
                                file_name = new_filename,
                                file_path = public_url,
                                file_size = uploaded_file.size,
                                file_type= uploaded_file.type
                            )
                st.success("Dataset successfully uploaded!")
                st.write(f"storage link: {public_url}")

            except SQLAlchemyError as db_error:
                st.error(f"Database error: {db_error}")

            except Exception as error:
                st.error(f"Something goind wrong-> {error}")

else:
    st.error("Please! first enter your email!")


if df is not None and not df.empty and dataset_id:
    profile = dataset_profiling_Engine(df)

    with engine.connect() as conn:
        query = """INSERT INTO dataset_profiles(dataset_id,row_count,column_count,total_missing,total_duplicate,memory_usage,created_at)
                    VALUES(:dataset_id,:row_count,:column_count,:total_missing,:total_duplicate,:memory_usage,:created_at)
        """

        conn.execute(text(query),{
            "dataset_id":int(dataset_id),
            "row_count":int(profile['row_count']),
            "column_count":int(profile['column_count']),
            "total_missing":int(profile['total_missing']),
            "total_duplicate":int(profile['total_duplicate']),
            "memory_usage":int(profile['memory_usage']),
            "created_at":datetime.datetime.utcnow()
        })

        conn.commit()

    st.subheader("Data profile:")
    st.write(f"Rows in dataset : {profile['row_count']}")
    st.write(f"Columns in dataset: {profile['column_count']}")
    st.write(f"Total missing value: {profile['total_missing']}")
    st.write(f"Total duplicated value: {profile['total_duplicate']}")
    st.write(f"Memory usage: {profile['memory_usage']}")
    
    
    schema = schema_detected(df)
    schema["amount"] = user_column_detector(df,AMOUNT_COLUMNS)
    schema["balance"] = user_column_detector(df,BALANCE_COLUMNS)
    schema["time"] =  user_column_detector(df,TIME_COLUMNS)
    schema["location"] = user_column_detector(df,LOCATION_COLUMNS)
    schema["devices"] = user_column_detector(df,DEVICE_COLUMNS)
    schema["network"] = user_column_detector(df,NETWORK_COLUMNS)
    schema["transaction"] = user_column_detector(df,TRANSACTION_TYPE_COLUMNS)
    schema["customer"] = user_column_detector(df,CUSTOMER_COLUMNS)
    schema["behaviour"] = user_column_detector(df,BEHAVIOUR_COLUMNS)
    schema["text"] = user_column_detector(df,TEXT_COLUMNS)
    schema["fraud"] = user_column_detector(df,FRAUD_COLUMNS)


    amount = schema['amount']
    balance = schema['balance']
    time = schema['time']
    location = schema['location']
    devices = schema['devices'] 
    network = schema['network']
    transaction = schema['transaction']
    customer = schema['customer']
    behaviour = schema['behaviour']
    taxt = schema['text']

    if amount:
        st.write(f"Amount detected -> {schema['amount'][0]} ! ")
    
    else:
        st.warning("Amount was not detected please check dataset!")

    if balance:
        st.write(f"Balance detected -> {schema['balance'][0]}")
    
    else:
        st.warning("Balance was not detected please check dataset!")
    
    if time:
        st.write(f"Time detected -> {schema['time'][0]}")
    
    else:
        st.warning("Time was not detected please check dataset!")
    
    if location:
        st.write(f"Location was detected -> {schema['location']}")
    
    else:
        st.warning("Location was not detected please check dataset!")
    
    if devices:
        st.warning(f"Devices detected -> {schema['devices']} ")
    
    else:
        st.warning("Devices was not detected please check dataset!")
    
    if network:
        st.write(f"Network detected -> {schema['network']}")
    
    else:
        st.warning("Network was not detected please check dataset!")
    
    if transaction:
        st.write(f"Transaction detected -> {schema['transaction']}")

    else:
        st.warning("Transaction was not detected please check dataset!")
    
    if customer:
        st.write(f"Customer detected -> {schema['customer']}")

    else:
        st.warning("Customer was not detected please check dataset!")
    
    if behaviour:
        st.write(f"Behaviour detected -> {schema['behaviour']}")
    
    else:
        st.warning("Behaviour was not detected please check dataset!")
    
    if taxt:
        st.write(f"Text was detected -> {schema['text']}!")

    else:
        st.warning("Taxt was not detected please check dataset!")
    
    normal_val,fraud_val = fraud_detect_col(df,schema)

    combo = schema["fraud"]
    typepo = ["Fraud","Normal"]

    visualization(df,schema,normal_val,fraud_val)

    st.title("Visualization")

    df =  feature_engineering(df,schema,fraud_val)

    df = preprocessing(df,schema)

    ml, X , X_test, Y_test = model(df,schema,fraud_val)

    df = insights(ml,X,X_test,Y_test,schema,df,fraud_val)

    df = insight_behavioral_baseline(df,schema)

    df = insight2(df,schema)
