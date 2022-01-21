from flask import Flask, jsonify,request
import pandas as pd
import pickle
import numpy as np
import pandas as pd 
from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings("ignore", category=UserWarning)



def load_data():
    
    #df=pd.read_csv('train_ech.csv',index_col='SK_ID_CURR',encoding ='utf-8')
    df=pd.read_feather('train_ech')
    df.index=df['SK_ID_CURR']
    X=df.copy()
    X.drop(columns=['TARGET'],inplace=True)    
    return df,X

def load_model():
    #lgbm = joblib.load("model_fin.joblib")
    
    pickle_in = open('model_lgbm.pkl', 'rb')
    lgbm = pickle.load(pickle_in)
    return lgbm

def load_prediction(df,id, model):  
    
    score_client=model.predict_proba(df[df.index == int(id)])[:,1]        
    return  score_client

# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'NOSECRETKEY'

df,X= load_data()
model= load_model()
ids_clients = df.index.values


@app.route('/')
def index():
    return 'test app_flask'

@app.route('/credit/<id_client>', methods=['GET'])
def credit(id_client):
    #récupération id client depuis argument url
    #id_client = request.args.get('id_client', default=1, type=int)
    #calculer prédiction de la probabilité de défaut
    prediction = load_prediction(X, id_client, model)
    
    # renvoyer la prediction 
    dict_final = {
        'proba' : prediction[0]
        }
    return jsonify(dict_final)

    

#lancement de l'application
if __name__ == '__main__':
    app.run(debug=True)





































