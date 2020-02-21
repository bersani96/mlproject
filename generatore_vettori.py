import glob
import json
import csv
import instaloader
from instaloader import Post

# Parametri 
PATH_PUBBLICITARIE="D:\\Documenti\\Università\\Progetto ML SII\\dataset\\nicolecarlsonxo\\pubblicitarie\\"
PATH_NON_PUBBLICITARIE=""
DEST="D:\\Documenti\\Università\\Progetto ML SII\\"

# ==== CONFIGURAZIONE VETTORE ===========
P_PAGINE_TAGGATE = 0 # Numero di pagine taggate nella didascalia
P_PAGINE_TAGGATE_FOTO = 1 # Numero di pagine taggate nella foto
P_BUSINESS_ACCOUNT = 2 # Se è un account business

# Restituisce il file del json corrispondente
def getJsonName(img):
    return img.split("UTC")[0] + "UTC.json"

# Restituisce il numero di pagine taggate nella didascalia
def getTagDidascalia(data):
    didascalia = data['node']['edge_media_to_caption']['edges'][0]['node']['text']
    return didascalia.count("@")

# Restituisce il numero di pagina taggate nella foto
def getTagFoto(data):
    post = Post.from_shortcode(L.context,"B8zEcEMIXIJ")
    return len(post.tagged_users)

# Restituisce 1 se è un account business, 0 altrimenti
def getBusinessAccount(data):
    if data['node']['owner']['is_business_account'] == True:
        return 1
    else:
        return 0

# Inizio script
L = instaloader.Instaloader()
# Prendo tutte le immagini nella directory
files=[f for f in glob.glob(PATH_PUBBLICITARIE+"*.jpg")]

#Prova
files=files[0:2]

# Apro il file csv di destinazione (dataset)
try:
    output_file= open(DEST+'vettori.csv', mode='w', newline='')
except:
    print("Errore nell'apertura del file di output.")
output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

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
        break
    data = json.load(json_file)
    
    # Controllo pagine taggate nella didascalia
    vet.append(getTagDidascalia(data))

    # Controllo delle pagine taggate nella foto
    vet.append(getTagFoto(data))

    # Controllo se è un account verificato
    vet.append(getBusinessAccount(data))

    output_writer.writerow(vet)
    json_file.close()
    i=i+1
    
output_file.close()
print("File scritto.")