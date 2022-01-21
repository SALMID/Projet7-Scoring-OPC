import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
#import seaborn as sns
import pickle
from urllib.request import urlopen
import json
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
#from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
import shap


def main() :
    
    def gauge_plot(score):
    
        fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = score,
        mode = "gauge+number",
        title = {'text': "Score client"},
        delta = {'reference': 0.3},
        gauge = {'axis': {'range': [None, 100]},
             'steps' : [
                 {'range': [0, 30], 'color': "whitesmoke"},
                 {'range': [30, 1], 'color': "lightgray"}],
             'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 30}}))

        st.plotly_chart(fig)
        
     
    def plot_radars(data):
    
        fig2 = go.Figure()

        trace0=fig2.add_trace(go.Scatterpolar(
        r=data[data["TARGET"]==2].iloc[:,0:].values.reshape(-1),
        theta=data.columns[0:6],
        fill='toself',
        name="Client sélectionné"
        ))
    
        trace1=fig2.add_trace(go.Scatterpolar(
        r=data[data["TARGET"]==1].iloc[:,0:].values.reshape(-1),
        theta=data.columns[0:6],
        fill='toself',
        name="Moyennes des clients similaires avec défaut de paiement"
        ))
    
        trace2=fig2.add_trace(go.Scatterpolar(
        r=data[data["TARGET"]==0].iloc[:,0:].values.reshape(-1),
        theta=data.columns[0:6],
        fill='toself',
         name="Moyennes des clients similaires sans défaut de paiement"
        ))
        data = [trace0, trace1]
    
        fig2.update_layout(
        polar=dict(
        radialaxis=dict(
          visible=True
          #range=[0, 1]
        )),
        legend=dict(
        yanchor="top",
        y=-0.1,
        xanchor="left",
        x=0.01
        ),
        title={'text': "Comparatif du client avec des clients similaires",
                'y':0.95,
                'x':0.5,
                'yanchor': 'top'},
        title_font_color="blue",
        title_font_size=17)
        st.plotly_chart(fig2)
        
    def bar_plot(df, col):
        labels={'CODE_GENDER': 'Genre',
            'NAME_EDUCATION_TYPE': "Niveau d'éducation",
            'NAME_INCOME_TYPE' : "Type de revenus",
            'NAME_CONTRACT_TYPE' : 'Types de contrats de crédits'
            }
    
        fig3 = px.bar(df, x=col, y="Percentage",
                 color="Cible",
                 labels=labels,
                 color_discrete_sequence=['#8eb5b5', '#b48fa2'],
                 text="Percentage"                 
                )
        fig3.update_layout(
            polar=dict(
            radialaxis=dict(
            visible=True       
            )),
            showlegend=True,
            title={'text': f"Répartition des impayés selon le {str.lower(labels[col])}",
                'y':0.95,
               'x':0.5,
               'yanchor': 'top'},
            title_font_color="blue",
            title_font_size=17)               
        st.plotly_chart(fig3) 
        
    def bar_plot2(df, col):
        labels={'CODE_GENDER': 'Genre',
            'NAME_EDUCATION_TYPE': "Niveau d'éducation",
            'NAME_INCOME_TYPE' : "Type de revenus",
            'NAME_CONTRACT_TYPE' : 'Types de contrats de crédits'
            }
    
        fig4 = px.bar(df, x=col, y="Percentage",
                 color="Cible",
                 labels=labels,
                 color_discrete_sequence=['#8eb5b5', '#b48fa2'],
                 text="Percentage"                 
                )
        fig4.update_layout(
            polar=dict(
            radialaxis=dict(
            visible=True       
            )),
            showlegend=True,
            title={'text': f"Répartition des impayés selon le {str.lower(labels[col])},chez les clients similaires au client sélectionné",
               'yanchor': 'top'},
            title_font_color="blue",
            title_font_size=17)               
        st.plotly_chart(fig4) 
        
    def percent(col):
        df=clients_sim.groupby(by=["TARGET",col]).agg({'SK_ID_CURR': 'count'}).reset_index()
        df['Percentage'] = 0
        for cat in df[col].unique():
            somme = df.groupby([col]).sum().loc[cat, 'SK_ID_CURR']
            df['Percentage'] = np.where(df[col] == cat,
                                    round(df['SK_ID_CURR'] / somme * 100, 1),
                                    df.Percentage)    
        df['Cible'] = 0
        df['Cible'] = np.where(df.TARGET == 1.0, "Clients avec impayés", df.Cible)
        df['Cible'] = np.where(df.TARGET == 0.0, "Clients sans impayés", df.Cible)
        return df
    
    @st.cache(suppress_st_warning=True) #mise en cache de la fonction pour exécution unique
    def load_data():    
        #data=pd.read_csv('Données/train_ech.csv',index_col=0,encoding ='utf-8')
        data=pd.read_feather("train_ech")
        feats = [f for f in data.columns if f not in ['SK_ID_CURR','TARGET']]
        data.drop(columns=['TARGET'],inplace=True) 
        df_nn_target=pd.read_csv('df_nn_target.csv',index_col=0,encoding ='utf-8')
        df_compa=pd.read_csv('df_compa_quali.csv',index_col=0,encoding ='utf-8')
        genre=pd.read_csv('genre')
        education=pd.read_csv('education')
        revenus=pd.read_csv('type_revenus')          
        return data,feats,df_nn_target,df_compa,genre,education,revenus
    
    @st.cache(suppress_st_warning=True,allow_output_mutation=True) #mise en cache de la fonction pour exécution unique
    def load_model():     
        pickle_in = open('model_lgbm.pkl', 'rb')
        model= pickle.load(pickle_in)
        
        nn = open('NearestNeighborsModel.pkl', 'rb')
        nn = pickle.load(nn)
        
        explainer=open('shap_values.pkl', 'rb')
        explainer=pickle.load(explainer)
        
        shap_values=open('shap_values.pkl', 'rb')
        shap_values=pickle.load(shap_values)        
        return model,nn,shap_values,explainer
    
   
    def var_interpretabilte (table):
        table['Age']=table.apply(lambda x: int((x["DAYS_BIRTH"]/-365)),axis=1)
        table['Durée d emploi (en années)']=table.apply(lambda x: round((x["DAYS_EMPLOYED"]/-365),1),axis=1)
        cols=['AMT_PAYMENT','CNT_PAYMENT','AMT_CREDIT','AMT_ANNUITY']
        for col in cols:
                table[col]=table.apply(lambda x: round(x[col],1),axis=1)
        
        table.drop(columns=['DAYS_BIRTH','DAYS_EMPLOYED'],inplace=True)
        
        table=table[['SK_ID_CURR','Age','Durée d emploi (en années)','AMT_CREDIT',
                    'CNT_PAYMENT','AMT_PAYMENT','AMT_ANNUITY','ANNUITY_INCOME_PERC','TARGET']]
    
        table.rename(columns={
    
                'AMT_PAYMENT':'Remboursements crédit précédent(en$)',
                'AMT_CREDIT':"Montant crédit précédent(en$)",
                'CNT_PAYMENT':"Durée du crédit précédent",
                'AMT_ANNUITY':'Annuités emprunt',
                'ANNUITY_INCOME_PERC':"Annuités/Revenus"},inplace=True)    
        return(table)
    
    
    
    data,feats,df_nn_target,df_compa,genre,education,revenus= load_data()
    
    model, nn,shap_values,explainer= load_model()
    

    #############################################################################################################################
    ### Title
    
    html_temp = """
    <div style="background-color: LightSlateGray; padding:5px; border-radius:8px">
    <h1 style="color: white; text-align:center">Évaluation des demandes de Crédit</h1>
    </div>
    <p style="font-size: 15px; font-style: italic; text-align:center">Pour aide à la décision</p>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    ##############################################################################################################################
    ### Sidebar
    st.sidebar.title("Menu")
    
    list=pd.read_csv('Données/df_nn_target.csv',index_col='SK_ID_CURR',encoding ='utf-8')
    identifiant=list.index.values
    
    identifiant = st.sidebar.selectbox('Choisir un Identifiant client: ', identifiant)
    
    
    sidebar_selection = st.sidebar.radio(
        ' ',
        ['Score de solvabilité client', 'Comparaison des clients'],
        )

    if sidebar_selection == 'Score de solvabilité client':
            
        with st.spinner('Calcul en cours'):
                     
            #API_url = "http://127.0.0.1:5000/credit/" + str(identifiant)
            API_url = "https://dsalmi-app-flask.herokuapp.com/credit/" + str(identifiant)

        with st.spinner('Chargement des résultats...'):
            json_url = urlopen(API_url)

            API_data = json.loads(json_url.read())
            classe_predite = API_data['proba']

            
            if classe_predite>0.3 :
                etat='Client à risque'
            else:
                 etat='Client peu risqué'            
            
            score=round(classe_predite*100)           
        
            chaine= etat +  ' avec ' + str(round(classe_predite*100)) +'% de risque de défaut de paiement'            
            
            html_temp3 = """
            <p style="font-size: 15px; font-style: italic">Décision avec un seuil de 30%</p>
            """         

        #affichage de la prédiction
            st.write('## Prédiction')
            gauge_plot(score)                
            st.markdown(chaine)
            st.markdown(html_temp3, unsafe_allow_html=True)
# ###################################################################################
            #Feature importance / description                
            #shap.summary_plot(shap_values, features=df_clt, feature_names=df_clt.columns, plot_type='bar')
                         
            st.write('## Interprétabilité du résultat')                            
            shap.initjs()   
            X=data[data['SK_ID_CURR']==identifiant]
            
            fig, ax = plt.subplots(figsize=(10, 10))
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            shap.summary_plot(shap_values, features=X, plot_type ="bar", max_display=10, color_bar=False, plot_size=(10, 10))            
            #shap.bar_plot(shap_values[0],feature_names=np.array(feats),max_display=10)            
            st.pyplot(fig)              
                
                    
                
                
# ##########################################################################################################        
                

    if sidebar_selection == 'Comparaison des clients': 
        
        list_graph=['Genre', "Niveau d'éducation","Types de revenus"]
        affichage_graph= st.sidebar.selectbox("Afficher d'autres graphiques: ", list_graph)
        html_temp8 = """
            <b style="font-size: 20px">Informations client : </b>
            """ 
        st.markdown(html_temp8, unsafe_allow_html=True)
        #st.write("Informations client : ")                
        
               
        interpretable_important_data_target = ['SK_ID_CURR',
                                       'AMT_PAYMENT',
                                        'AMT_ANNUITY',
                                        'AMT_CREDIT',
                                        'CNT_PAYMENT',
                                        'DAYS_BIRTH',
                                        'DAYS_EMPLOYED',
                                        'ANNUITY_INCOME_PERC', 
                                        'EXT_SOURCE_1',
                                        'EXT_SOURCE_2',
                                        'EXT_SOURCE_3',
                                       'TARGET']
        
        # plus proches voisins
        
        
        df_nn =df_nn_target.iloc[:,:-1]
        df_client=df_nn[df_nn['SK_ID_CURR'] == identifiant]
        std=StandardScaler()
        client_list = std.fit_transform(df_client)  # standardisation

        distance, voisins = nn.kneighbors(client_list)
        voisins = voisins[0]
        # on crée un dataframe avec les voisins
        voisins_table = pd.DataFrame()
        for v in range(len(voisins)):
            voisins_table[v] = df_nn.iloc[voisins[v]]

        tab=pd.DataFrame(voisins_table.iloc[0,:])
        tab['SK_ID_CURR']=tab.apply(lambda x: int(x['SK_ID_CURR']),axis=1)
        clt_sim=pd.merge(tab,df_nn_target,how='left',on='SK_ID_CURR')

        clt_sim=var_interpretabilte(clt_sim)
        stat=clt_sim.groupby(by=["TARGET"],as_index=False).agg(
            {'Remboursements crédit précédent(en$)': 'mean',
             'Annuités emprunt': 'mean',
            "Montant crédit précédent(en$)": 'mean',
            "Durée du crédit précédent":'mean',
            'Age':'mean',
            'Durée d emploi (en années)':'mean',
            'Annuités/Revenus':'mean'})

        df_client=df_nn_target[df_nn_target['SK_ID_CURR'] == identifiant]

        df_client=var_interpretabilte(df_client)
        df_client["TARGET"]=2
        df_comparatif=pd.concat([stat,df_client])
        df_comparatif.drop(columns=['SK_ID_CURR'],inplace=True)

        data_scal = df_comparatif.iloc[:,1:7]
        scaler= StandardScaler()
        data_scaled= scaler.fit_transform(data_scal)
        df_comparatif_scaled = pd.DataFrame(data_scaled,columns=df_comparatif.iloc[:,1:7].columns)

        X_temp = df_comparatif[['TARGET']]
        X_temp.reset_index(inplace=True)
        df_comparatif_scaled= pd.merge(df_comparatif_scaled, X_temp, left_index=True, right_index=True)
        
        clients_sim=pd.merge(tab,df_compa,how='left',on='SK_ID_CURR')
        client=df_compa[df_compa['SK_ID_CURR'] == identifiant]
        
        st.table(df_client.iloc[:,1:7])        
        plot_radars(data=df_comparatif_scaled )       
             
        html_temp4 = """
            <b style="font-size: 20px">Le client est une femme.</b>
            """ 
        html_temp5 = """
            <b style="font-size: 20px">Le client est un homme.</b>
            """ 
        html_temp6 = """
            <b style="font-size: 20px">Niveau d'éducation du client :</b>
            """ 
        html_temp7 = """
            <b style="font-size: 20px">Type de revenus du client : </b>
            """ 
        
        
        if affichage_graph=="Genre":
            
            cat = df_compa[df_compa['SK_ID_CURR'] == identifiant]['CODE_GENDER'].iloc[0]                   
            if cat == 'F':
                    #st.write('Le client est une femme.')
                    st.markdown(html_temp4, unsafe_allow_html=True)
            else:
                    #st.write('Le client est un homme.')
                    st.markdown(html_temp4, unsafe_allow_html=True)
                    
            bar_plot(genre, 'CODE_GENDER')        
        
        if affichage_graph=="Niveau d'éducation":
            cat = df_compa[df_compa['SK_ID_CURR'] == identifiant]['NAME_EDUCATION_TYPE'].iloc[0]
            #st.write("Niveau d'éducation du client : " + cat)
            st.markdown(html_temp6 + cat, unsafe_allow_html=True)
            bar_plot(education, 'NAME_EDUCATION_TYPE')
            
        if affichage_graph=="Types de revenus":
            cat = df_compa[df_compa['SK_ID_CURR'] == identifiant]['NAME_INCOME_TYPE'].iloc[0]
            st.markdown(html_temp7 + cat, unsafe_allow_html=True)
            #st.write("Type de revenus du client : " + cat)
            bar_plot(revenus, 'NAME_INCOME_TYPE')                
            
        if affichage_graph=="Genre":
            genre_sim=percent('CODE_GENDER')
            bar_plot2(genre_sim, 'CODE_GENDER')
            
        if affichage_graph=="Niveau d'éducation":
            genre_sim=percent('NAME_EDUCATION_TYPE')
            bar_plot2(genre_sim, 'NAME_EDUCATION_TYPE')
            
        if affichage_graph=="Types de revenus":
            genre_sim=percent('NAME_INCOME_TYPE')
            bar_plot2(genre_sim, 'NAME_INCOME_TYPE')        
        
    
if __name__ == '__main__':
    main()    
    
    
    
    
    
    
    
   
