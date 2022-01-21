import requests
import signal
from piou_piou_raoul_aurelie_dao import *
from piou_piou_raoul_aurelie_objets import *
from datetime import datetime
import time


# ---------------------------------------------------------------------------------------------
#                               CONSTANTES
# ---------------------------------------------------------------------------------------------
stations_proche_nantes = {
    "Groix": 298,
    "Brétignolles sur mer": 113,
    "L'île-d'Olonne":308,
    "La Rochelle":565
}

stations_22 = {
    "Pordic": 194,
    "Plérin": 116,
    "Champeaux":334
}

BDD_NAME = 'my_piou_piou_raoul_aurelie.db'
PP_URL_API_LIVE = "http://api.pioupiou.fr/v1/live/"
TIMEOUT = 5 # nombre de seconde à attendre
SLEEP_TIME =  30 # in secondes
# Récupérer les stations de la bdd :
gestionnaire = GestionnaireDeStations()

curent_path = getcwd()+ "\\simplon\\projets\\projet_sql\\"

verbose = 1
# ---------------------------------------------------------------------------------------------
#                               FONCTIONS
# ---------------------------------------------------------------------------------------------

def station_information(url, id_station, verbose=False):
    """Appelle l'API pour récupérer la mesure de la station reçue

    Args:
        url (str): url de l'API (sans l'id de la station)
        station (Station): Station à mettre à jour

    Raises:
        Exception: En cas d'incohérence entre les donnes reçues et la station
        http_error : En cas d'erreur d'accès à l'API

    Returns:
        Station: la station ou None
    """
    nouvelle_station = None
    if id_station is not None and isinstance(id_station, int):       
        # Récupération des données de l'API avec l'ID de la station
        resp = requests.get(url=url+str(id_station))
        data = resp.json()
        # Vérification du code réponse
        if resp.status_code == 200 :
            if verbose>1:
                print(f"API > Statut {resp.status_code}")
            # Traitement de la réponse JSON
            data = data['data']
            # Information sur la station
            id = data['id']
            name = data['meta']['name']
            longitude = data['location']['longitude']
            latitude = data['location']['latitude']
            
            nouvelle_station = Station(id, name, latitude, longitude)
            if verbose:
                    print(f"{nouvelle_station}")
        else:
            print(f'{resp.status_code} => {data["error_code"]} : {data["error_message"]}')
            resp.raise_for_status()
    return nouvelle_station


def mesure_courante_pour_la_station(url, station, verbose=False):
    """Appelle l'API pour récupérer la mesure de la station reçue

    Args:
        url (str): url de l'API (sans l'id de la station)
        station (Station): Station à mettre à jour

    Raises:
        Exception: En cas d'incohérence entre les donnes reçues et la station
        http_error : En cas d'erreur d'accès à l'API

    Returns:
        Mesure: la mesure ou None
    """
    mesure = None
    if station is not None and isinstance(station, Station):       
        # Récupération des données de l'API avec l'ID de la station
        resp = requests.get(url=url+str(station.id))
        data = resp.json()
        # Vérification du code réponse
        if resp.status_code == 200 :
            if verbose>1:
                print(f"API > Statut {resp.status_code}")
            # Traitement de la réponse JSON
            data = data['data']
            # Information sur la station
            id = data['id']
            name = data['meta']['name']
            # Vérification de la cohérence entre la station de la mesure et la station courante
            if id == station.id:
                # Information sur la mesure
                data = data['measurements']
                measures_date_str1 = data['date']
                # convertion en date
                measures_date_str = datetime.fromisoformat(measures_date_str1[:-1])
                # "2022-01-17T09:45:47.000Z"  ==> YYYY-MM-DDTHH:MM:SS.mmmmmm
                measures_date = measures_date_str.strftime('%Y-%m-%d %H:%M:%S')
                wind_heading = data['wind_heading']
                wind_speed_avg = data['wind_speed_avg']
                wind_speed_min = data['wind_speed_min']
                wind_speed_max = data['wind_speed_max']
                if verbose:
                    print(f"{id} : {measures_date} - {wind_heading}, {wind_speed_avg}, {wind_speed_min} => {name}")
                mesure = station.ajouter_mesure(date=measures_date, wind_heading=wind_heading, wind_speed_avg=wind_speed_avg, wind_speed_max=wind_speed_max, wind_speed_min=wind_speed_min)
            else:
                raise Exception(f"La mesure reçue pour la station {id}-{name} ne concerne pas la station {station.id}-{station.name}")
        else:
            print(f'{resp.status_code} => {data["error_code"]} : {data["error_message"]}')
            resp.raise_for_status()
    return mesure
   

def recuperer_mesures(url, gestionnaire, verbose=False):

    mesures_ajoutees = []
    # Récupéreration des mesures pour chaque station
    for station in gestionnaire.stations:
        mesure = mesure_courante_pour_la_station(url, station, verbose)
        if mesure is not None:
            nouvelle_mesure = station.ajouter_mesure(mesure)
            if nouvelle_mesure != mesure:
                # Sauvegarder en BDD
                mesures_ajoutees.append( ma_dao.ajouter_mesure(nouvelle_mesure))
    return mesures_ajoutees


def proposition_arret_programme():      
    try:
        foo = input(f'Vous avez {TIMEOUT} seconde pour quitter en tapant "exit" :')
        return foo
    except:
        # timeout
        return
   
def nouvelle_mise_jour(signum, frame):
    # Récupéreration des mesures pour chaque station
    mesures_ajoutees = recuperer_mesures(PP_URL_API_LIVE, gestionnaire, verbose)

# ---------------------------------------------------------------------------------------------
#                               MAIN
# ---------------------------------------------------------------------------------------------



ma_dao = PiouPiouDao(curent_path+BDD_NAME)
if ma_dao.initialiser_bdd(verbose=verbose):
    nb_stations = ma_dao.nombre_stations(verbose=verbose)
    if nb_stations == 0:
        # Création des stations en BDD
        for id in stations_proche_nantes.values():  
            # TODO : récupérer les informations de la station via l'API
            station = station_information(PP_URL_API_LIVE, id, verbose)
            if station is not None:
                res = ma_dao.ajouter_station(station)
                if res != id:
                    print(f"SQLite > La station {id} soit {station} n'a pas pu être insérée en BDD.")
            else:
                print(f"API > Aucune information récupérée sur la station {id}.")
    
    # Normalement ici il y a des stations en BDD
    nb_stations = ma_dao.nombre_stations(verbose=verbose)
    if nb_stations == 0:
        raise Exception("Aucune station.")
    
    
    station_rows = ma_dao.stations()
    if station_rows is not None:
        for row in station_rows:
            nouvelle_station = Station(station_rows[0], station_rows[1], station_rows[2], station_rows[3])
            # Ajout de la nouvelle station dans le gestionnaire
            gestionnaire.stations = nouvelle_station
    
    # TODO : à voir si on veut récupérer les mesures déjà en BDD ou non.

    # Boucler pour mise à jour régulière
    
    # signal.signal(signal.SIGALRM, nouvelle_mise_jour)
    sortie = False    
    while not sortie:
        # set alarm
        # signal.alarm(TIMEOUT)
        s = proposition_arret_programme()
        time.sleep(SLEEP_TIME)
        sortie = "exit" in s
        # disable the alarm after success
        # signal.alarm(0)

    print(f"---------------------- END ----------------------")
    

