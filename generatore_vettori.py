# Da caricare sul drive, nella stessa directory del colab

import glob
import json
import csv
import instaloader
from instaloader import Post
from yolo_model import YoloModel

# Parametri 
# PATH_PUBBLICITARIE="D:\\Documenti\\Università\\Progetto ML SII\\dataset\\nicolecarlsonxo\\pubblicitarie\\"
# PATH_NON_PUBBLICITARIE=""
# DEST="D:\\Documenti\\Università\\Progetto ML SII\\"

# ==== CONFIGURAZIONE VETTORE ===========
# P_PAGINE_TAGGATE = 0 # Numero di pagine taggate nella didascalia
# P_PAGINE_TAGGATE_FOTO = 1 # Numero di pagine taggate nella foto
# P_BUSINESS_ACCOUNT = 2 # Se è un account business
# P_GEOLOC = 3 # Se il post è geolocalizzato

# Restituisce il file del json corrispondente
def getJsonName(img):
    return img.split("UTC")[0] + "UTC.json"

# Restituisce il numero di pagine taggate nella didascalia
def getTagDidascalia(data):
    didascalia=""
    try: 
        didascalia = data['node']['edge_media_to_caption']['edges'][0]['node']['text']
    except:
        pass
    return didascalia.count("@")

# Restituisce il numero di pagina taggate nella foto
def getTagFoto(data, loader):
    post = Post.from_shortcode(loader.context,"B8zEcEMIXIJ")
    return len(post.tagged_users)

# Restituisce 1 se è un account business, 0 altrimenti
def getBusinessAccount(data):
    if data['node']['owner']['is_business_account'] == True:
        return 1
    else:
        return 0

# Restituisce l'output del modello di object detection 
def getOggetti(img,model) :
    return model.detect(img)

# Restituisce se il post è localizzato o meno (1 se localizzato, 0 altrimenti)
def getLocalizzato(data):
    if 'location' in data['node']:
        return 1
    else:
        return 0

def generaVettori(path_pubblicitarie, path_non_pubblicitarie, path_destinazione, base_path_yolo):

    L = instaloader.Instaloader()
    model = YoloModel(base_path_yolo)

    # Apro il file csv di destinazione (dataset)
    try:
        output_file= open(path_destinazione+'vettori.csv', mode='w', newline='')
    except:
        print("Errore nell'apertura del file di output.")
        exit()

    output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    # Prima riga: feature names
    vet=["Pag. taggate nella didascalia", "Pag. taggate nella foto", "Business account", "Geolocalizzazione", "Pubblicitario"]
    output_writer.writerow(vet)

    # Due iterazioni: pubblicitarie e non pubblicitarie
    for j in range(0,2): 
        if j==0:
            # Pubblicitarie
            print("Analizzo le immagini pubblicitarie in " + path_pubblicitarie)
            path=path_pubblicitarie
        else:
            # Non pubblicitarie
             print("Analizzo le immagini non pubblicitarie in " + path_non_pubblicitarie)
             path=path_non_pubblicitarie

        # Prendo tutte le immagini nella directory
        files=[f for f in glob.glob(path+"*.jpg")]

        #Prova con pochi input
        files=files[0:2]

        i=0
        for img in files:
            print("Immagine " + str(i))
            vet=[] # Riga da scrivere nel file
            filename=getJsonName(img)
            print("\t File: " + filename)
            # Apro il json corrispondente alla foto analizzata
            try:
                json_file= open(filename)
            except:
                print("Errore nell'apertura del file")
                continue # Passo al prossimo item
            data = json.load(json_file)
            
            # Controllo pagine taggate nella didascalia
            vet.append(getTagDidascalia(data))

            # Controllo delle pagine taggate nella foto
            vet.append(getTagFoto(data,L))

            # Controllo se è un account verificato
            vet.append(getBusinessAccount(data))

            # Controllo se il post è geolocalizzato
            vet.append(getLocalizzato(data))

            # TODO: Da finire!
            # Controllo gli oggetti all'interno della foto
            boxes, scores, classes, nums = getOggetti(img,model)
            print("Oggetti nella foto: " + str(nums[0]))

            # Aggiungo etichetta
            vet.append(j)

            output_writer.writerow(vet)
            json_file.close()
            i=i+1
        
    output_file.close()
    print("File scritto in " + path_destinazione)