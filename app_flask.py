#APP FLASK (commande : flask run)
# Partie formulaire non utilisée (uniquement appel à l'API)

from flask import Flask, jsonify
import pandas as pd
import pickle

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def load_data():
    '''loading the train and original data'''
    data = pd.read_feather('train_ech')
    data.index=data['SK_ID_CURR']
    target = data['TARGET']
    return data, target

def load_model():
    '''loading the trained model'''
    pickle_in = open('model/model_lgbm.pkl', 'rb')
    clf_x = pickle.load(pickle_in)
    return clf_x

def load_prediction(sample, id, clf):
    '''Predict the default probability for a client using SK_CURR_ID'''
    # X=sample.iloc[:, :-1]
    X=df.copy()
    score = clf.predict_proba(X[X.index == int(id)])[:,1]
    return score

# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'nosecretkey'

data, sample, target, description = load_data()
clf = load_model()
# clf_str_seq = json.dumps(clf)
ids_clients = data.index.values

@app.route('/')
def index():
    return 'app flask !!'

@app.route('/credit/<chk_id>', methods=['GET'])
def credit(chk_id):
    #récupération id client depuis argument url
    #chk_id = request.args.get('chk_id', default=1, type=int)
    #calculer prédiction de la probabilité de défaut
    prediction = load_prediction(X, chk_id, clf)
    # renvoyer la prediction au demandeur
    dict_final = {
        'proba' : float(prediction)
        }
    return jsonify(dict_final)

#lancement de l'application
if __name__ == '__main__':
    app.run(debug=True)



































