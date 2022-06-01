import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
import shap
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, RandomForestClassifier


from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2

#from sklearn.metrics import mean_squared_error, r2_score, f1_score

st.write('Hello, *World!* :sunglasses:')






st.write('''
# Application de Prévision des prix Airbnb
''')




st.write('''
# Dataframe utilisé
''')
#On importe la base de données
df = pd.read_csv("df_streamlit.csv", sep='\t')

#df = pd.read_csv("df_sansville.csv", sep='\t')
#df = df.drop(['Unnamed: 0','nb_comment'], axis=1)
df = df.drop(['Unnamed: 0'], axis=1)

st.write(df)
st.caption("0 = NC ou rien et 1 = présent")

st.write(df.shape)

#df = df.drop(['Unnamed: 0'],axis=1)
y = df.pop('price')
X = df

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42) # On choisit de prendre 20% de données pour le test

reg_rf = RandomForestRegressor(max_depth=5, random_state=0,
                             n_estimators=50)
forest = reg_rf.fit(X_train, y_train)

st.write(forest.score(X_train,y_train))

st.write(forest.score(X_test, y_test))


st.sidebar.header("Les parametres d'entrée")

def user_input():
  #  superhost = st.sidebar.selectbox('Superhost', ('0', '1'))  #VALEUR min puis max puis valeurs par def
  #  statut  = st.sidebar.selectbox('Statut', ('0','1'))
    #grade = st.sidebar.slider('Note', 3.0, 5.0, 4.0)
    voyageurs = st.sidebar.slider('Nombre de voyageurs', 1, 16, 1)
    nb_chambres = st.sidebar.slider('Nombre de chambres', 0, 11, 1)
    nb_lits = st.sidebar.slider('Nombre de lit', 1, 26, 1)
    nb_sdb = st.sidebar.slider('Nombre de salle de bain', 0, 12, 1)
    #city = st.sidebar.slider('La largeur du Petal', 1, 16, 3) #!!!!!! mettre toute les villes dispo
    #cuisine = st.sidebar.selectbox('Cuisine',('0','1'))
    #Télévision = st.sidebar.selectbox('Télévision',('0','1'))
   # Sèche_cheveux = st.sidebar.selectbox('Sèche-cheveux',('0','1'))
  #  Lave_vaisselle = st.sidebar.selectbox('Lave-vaisselle',('0','1'))
    #Eau_chaude = st.sidebar.selectbox('Eau chaude',('0','1'))
    #Draps = st.sidebar.selectbox('Draps',('0','1'))
   # Wifi = st.sidebar.selectbox('Wifi',('0','1'))
    #Climatisation = st.sidebar.selectbox('Climatisation',('0','1'))
   # Entrée_privée = st.sidebar.selectbox('Entrée privée',('0','1'))
    Jacuzzi = st.sidebar.selectbox('Jacuzzi',('0','1'))
    Barbecue = st.sidebar.selectbox('Barbecue',('0','1'))
   # Équipements_de_base = st.sidebar.selectbox('Équipements de base',('0','1'))
   # Cintres = st.sidebar.selectbox('Cintres',('0','1'))
   # Lave_linge = st.sidebar.selectbox('Lave-linge',('0','1'))
    #Détecteur_de_fumée = st.sidebar.selectbox('Détecteur de fumée',('0','1'))
    #Four_à_micro_ondes = st.sidebar.selectbox('Four à micro-ondes',('0','1'))
   # Réfrigérateur = st.sidebar.selectbox('Réfrigérateur',('0','1'))
   # Cuisinière = st.sidebar.selectbox('Cuisinière',('0','1'))
   # Cafetière = st.sidebar.selectbox('Cafetière',('0','1'))
   # Détecteur_de_monoxyde_de_carbone = st.sidebar.selectbox('Détecteur de monoxyde de carbone',('0','1'))
    #Séjours_longue_durée_autorisés = st.sidebar.selectbox('Séjours longue durée autorisés',('0','1'))
  #  Baignoire = st.sidebar.selectbox('Baignoire',('0','1'))
  #  #Fer_à_repasser = st.sidebar.selectbox('Fer à repasser',('0','1'))
  #  TV_avec_abonnement_standard_au_câble = st.sidebar.selectbox('TV avec abonnement standard au câble',('0','1'))
    Lit_pour_bébé = st.sidebar.selectbox('Lit pour bébé',('0','1'))
 #   Lit_parapluie = st.sidebar.selectbox('Lit parapluie',('0','1'))
  #  Accès_plage_ou_bord_de_mer = st.sidebar.selectbox('Accès plage ou bord de mer',('0','1'))
  #  Ascenseur = st.sidebar.selectbox('Ascenseur',('0','1'))
   # Stationnement_payant_à_l_extérieur_de_la_propriété = st.sidebar.selectbox("Stationnement payant à l'extérieur de la propriété",('0','1'))
   # Arrivée_autonome = st.sidebar.selectbox('Arrivée autonome',('0','1'))
  #  Logement_de_plain_pied = st.sidebar.selectbox('Logement de plain-pied',('0','1'))
  #  Vue_panoramique_sur_la_ville = st.sidebar.selectbox('Vue panoramique sur la ville',('0','1'))
 #   Vue_sur_la_piscine = st.sidebar.selectbox('Vue sur la piscine',('0','1'))
  #  Vue_sur_la_baie = st.sidebar.selectbox('Vue sur la baie',('0','1'))
   # Vue_sur_le_jardin = st.sidebar.selectbox('Vue sur le jardin',('0','1'))
   # Vue_sur_la_plage = st.sidebar.selectbox('Vue sur la plage',('0','1'))
  #  Vue_sur_le_parc = st.sidebar.selectbox('Vue sur le parc',('0','1'))
  #  Vue_sur_le_port = st.sidebar.selectbox('Vue sur le port',('0','1'))
  #  Vue_sur_la_mer = st.sidebar.selectbox('Vue sur la mer',('0','1'))
 #   Vue_sur_la_montagne = st.sidebar.selectbox('Vue sur la montagne',('0','1'))
 ##   Savon_pour_le_corps = st.sidebar.selectbox('Savon pour le corps',('0','1'))
    #Oreillers_et_couvertures_supplémentaires = st.sidebar.selectbox('Oreillers et couvertures supplémentaires',('0','1'))
 #   Connexion_Ethernet = st.sidebar.selectbox('Connexion Ethernet',('0','1'))
    Extincteur = st.sidebar.selectbox('Extincteur',('0','1'))
   # Congélateur = st.sidebar.selectbox('Congélateur',('0','1'))
   # Four = st.sidebar.selectbox('Four',('0','1'))
 #   Coffre_fort = st.sidebar.selectbox('Coffre-fort',('0','1'))
    Ustensiles_de_barbecue = st.sidebar.selectbox('Ustensiles de barbecue',('0','1'))
  #  Patio_ou_balcon : privé(e) = st.sidebar.selectbox('Patio ou balcon : privé(e)',('0','1'))
    Jardin_privé_et_Clôture_intégrale = st.sidebar.selectbox('Jardin privé(e), Clôture intégrale',('0','1'))
    Piscine = st.sidebar.selectbox('Piscine',('0','1'))
    Salle_de_sport = st.sidebar.selectbox('Salle de sport',('0','1'))
  #  Animaux_acceptés = st.sidebar.selectbox('Animaux acceptés',('0','1'))
  #  Dépôt_de_bagages_autorisé = st.sidebar.selectbox('Dépôt de bagages autorisé',('0','1'))
    #Clés_remises_par_l_hôte = st.sidebar.selectbox("Clés remises par l'hôte",('0','1'))
   # Équipements_de_cuisine_de_base = st.sidebar.selectbox('Équipements de cuisine de base',('0','1'))







#On récupere ces données dans un dico data
    data = {#'superhost':superhost,
               # 'statut':statut,
              #  'grade':grade,
                'voyageurs':voyageurs,
                'nb_chambres': nb_chambres,
                'nb_lits': nb_lits,
                'nb_sdb': nb_sdb,
               # 'cuisine': cuisine,
               # 'Télévision': Télévision,
             #   'Sèche_cheveux': Sèche_cheveux,
               # 'Lave_vaisselle': Lave_vaisselle,
            #    'Eau_chaude': Eau_chaude,
            #    'Wifi': Wifi,
            #    'Climatisation': Climatisation,
            #    'Entrée_privée': Entrée_privée,
                'Jacuzzi': Jacuzzi,
                'Barbecue': Barbecue,
             #   'Équipements_de_base': Équipements_de_base,
             #   'Cintres': Cintres,
             #   'Lave_linge': Lave_linge,
             #   'Détecteur_de_fumée': Détecteur_de_fumée,
            #    'Four_à_micro_ondes': Four_à_micro_ondes,
            #    'Réfrigérateur': Réfrigérateur,
            #    'Cuisinière': Cuisinière,
             #   'Cafetière': Cafetière,
              #  'Détecteur_de_monoxyde_de_carbone': Détecteur_de_monoxyde_de_carbone,
             #   'Séjours_longue_durée_autorisés': Séjours_longue_durée_autorisés,
           #     'Baignoire': Baignoire,
            #    'Fer_à_repasser': Fer_à_repasser,
              #  'TV_avec_abonnement_standard_au_câble': TV_avec_abonnement_standard_au_câble,
                 'Lit_pour_bébé': Lit_pour_bébé,
            #    'Lit_parapluie': Lit_parapluie,
            #    'Accès_plage_ou_bord_de_mer': Accès_plage_ou_bord_de_mer,
              #  'Ascenseur': Ascenseur,
                #'Stationnement_payant_à_l_extérieur_de_la_propriété': Stationnement_payant_à_l_extérieur_de_la_propriété,
              #  'Arrivée_autonome': Arrivée_autonome,
             #   'Logement_de_plain_pied': Logement_de_plain_pied,
            #    'Vue_panoramique_sur_la_ville': Vue_panoramique_sur_la_ville,
            #    'Vue_sur_la_piscine': Vue_sur_la_piscine,
             #   'Vue_sur_la_baie': Vue_sur_la_baie,
            #   'Vue_sur_le_jardin': Vue_sur_le_jardin,
            #    'Vue_sur_la_plage': Vue_sur_la_plage,
             #   'Vue_sur_le_parc': Vue_sur_le_parc,
            #    'Vue_sur_le_port': Vue_sur_le_port,
            #    'Vue_sur_la_mer': Vue_sur_la_mer,
            #    'Vue_sur_la_montagne': Vue_sur_la_montagne,
           #     'Savon_pour_le_corps': Savon_pour_le_corps,
         #       'Oreillers_et_couvertures_supplémentaires': Oreillers_et_couvertures_supplémentaires,
          #      'Connexion_Ethernet': Connexion_Ethernet,
                'Extincteur': Extincteur,
              #  'Congélateur': Congélateur,
              #  'Four': Four,
            #    'Coffre_fort': Coffre_fort,
                'Ustensiles_de_barbecue': Ustensiles_de_barbecue,
               # 'Patio_ou_balcon': Patio_ou_balcon,
                'Jardin_privé_et_Clôture_intégrale': Jardin_privé_et_Clôture_intégrale,
                'Piscine': Piscine,
                'Salle_de_sport': Salle_de_sport,
              #  'Animaux_acceptés': Animaux_acceptés,
              #  'Dépôt_de_bagages_autorisé': Dépôt_de_bagages_autorisé,
            #    'Clés_remises_par_l_hôte': Clés_remises_par_l_hôte,
           #     'Équipements_de_cuisine_de_base': Équipements_de_cuisine_de_base
            }
    location_params=pd.DataFrame(data, index=[0])
    return location_params



input_df = user_input()

for col in input_df.columns:
    input_df[col] = pd.to_numeric(input_df[col])



st.subheader('On veut trouver le prix de ce bien')

st.write(input_df)


nb_lits1 = input_df['nb_lits'][0]
voyageurs1 = input_df['voyageurs'][0]
nb_sdb1 = input_df['nb_sdb'][0]
nb_chambres1 = input_df['nb_chambres'][0]
Salle_de_sport1 = input_df['Salle_de_sport'][0]
Jacuzzi1 = input_df["Jacuzzi"][0]
Barbecue1 = input_df["Barbecue"][0]
Lit_pour_bébé1 = input_df["Lit_pour_bébé"][0]
Extincteur1 = input_df["Extincteur"][0]
Jardin1 = input_df["Jardin_privé_et_Clôture_intégrale"][0]
Piscine1 = input_df["Piscine"][0]
Ustensiles_de_barbecue1 = input_df["Ustensiles_de_barbecue"][0]






def prix(model, voyageurs=voyageurs1, nb_chambres=nb_chambres1,
       nb_lits=nb_lits1, nb_sdb=nb_sdb1, Jacuzzi=Jacuzzi1,
       Barbecue=Barbecue1, Lit_pour_bébé=Lit_pour_bébé1,Extincteur=Extincteur1,
        Ustensiles_de_barbecue=Ustensiles_de_barbecue1,
       Jardin=Jardin1, Piscine=Piscine1, Salle_de_sport=Salle_de_sport1):
    x = np.array([[voyageurs, nb_chambres,
       nb_lits, nb_sdb, Jacuzzi,Barbecue, Lit_pour_bébé,
        Extincteur,Ustensiles_de_barbecue,
       Jardin, Piscine, Salle_de_sport]]).reshape(1,12)
    st.write(model.predict(x))
    #st.write(nb_lits)
   # st.write(nb_sdb)
   # st.write(voyageurs)
   # st.write(nb_chambres)

st.set_option('deprecation.showPyplotGlobalUse', False)
st.write("Prix : ", prix(reg_rf))

explainer = shap.TreeExplainer(reg_rf)
shap_values = explainer.shap_values(X)

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')














