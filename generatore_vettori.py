# Da caricare sul drive, nella stessa directory del colab

import glob
import json
import csv
import instaloader
from instaloader import Post
from yolo_model import YoloModel
import numpy as np


class GeneratoreVettori():
    """
    Classe che modella un generatore di vettori di feature.

    Attributi:
        L (Instaloader): instaloader per ottenere dati e informazioni sul post instagram.
        model (YoloModel): modello di object detection.
    """

    def __init__(self, base_path_yolo):
        """
        Costruttore.

        Parametri:
            base_path_yolo (str): path della directory yolo. Deve terminare con /
        """
        self.L = instaloader.Instaloader()
        self.model = YoloModel(base_path_yolo)

    # ==== CONFIGURAZIONE VETTORE ===========
    # P_PAGINE_TAGGATE = 0 # Numero di pagine taggate nella didascalia
    # P_PAGINE_TAGGATE_FOTO = 1 # Numero di pagine taggate nella foto
    # P_BUSINESS_ACCOUNT = 2 # Se è un account business
    # P_GEOLOC = 3 # Se il post è geolocalizzato

    # P_SCORE_PERSONA = 4
    # P_X1_PERSONA = 5
    # P_Y1_PERSONA = 6
    # P_X2_PERSONA = 7
    # P_Y2_PERSONA = 8

    # P_SCORE_BORSA = 9
    # P_X1_BORSA = 10
    # P_Y1_BORSA = 11
    # P_X2_BORSA = 12
    # P_Y2_BORSA = 13

    # P_SCORE_OROLOGIO = 14
    # P_X1_OROLOGIO = 15
    # P_Y1_OROLOGIO = 16
    # P_X2_OROLOGIO = 17
    # P_Y2_OROLOGIO = 18

    def getJsonName(self, img):
        """
        Restituisce il nome del file json corrispondente alla foto.

        Parametri: 
            img (str): path dell'immagine
        """
        if "test_" in img:
            return img.split(".jpg")[0]+".json"      # Immagini di test 
        else:
            return img.split("UTC")[0] + "UTC.json"     # Immagini di training

    def getTagDidascalia(self,data):
        """
        Restituisce il numero di pagine taggate nella didascalia

        Parametri:
            data (dict): dizionario generato da file json che descrive il post instagram.
        """
        didascalia=""
        try: 
            didascalia = data['node']['edge_media_to_caption']['edges'][0]['node']['text']
        except:
            print("Errore: didascalia non trovata.")
        return didascalia.count("@")

    def getTagFoto(self, data, loader):
        """
        Restituisce il numero di pagina taggate nella foto

        Parametri:
            data (dict): dizionario generato da file json che descrive il post instagram.
            loader (Instaloader): instaloader per ottenere informazioni sul post instagram.
        """
        try:
            post = Post.from_shortcode(loader.context,data['node']['shortcode'])
        except:
            print("Errore: Post non trovato.")
            return 0
        return len(post.tagged_users)

    def getBusinessAccount(self, data):
        """
        Restituisce 1 se è un account business, 0 altrimenti

        Parametri:
            data (dict): dizionario generato da file json che descrive il post instagram.
        """
        try :
            if data['node']['owner']['is_business_account'] == True:
                return 1
            else:
                return 0
        except:
            return 0

    def getOggetti(self, img,model) :
        """
        Restituisce l'output del modello di object detection.

        Parametri:
            img (str): path dell'immagine su cui fare object detection.
            model (YoloModel): modello da usare.
        """
        return model.detect(img)

    def getLocalizzato(self, data):
        """
        Restituisce se il post è localizzato o meno (1 se localizzato, 0 altrimenti).

        Parametri:
            data (dict): dizionario generato da file json che descrive il post instagram.
        """
        if 'location' in data['node']:
            return 1
        else:
            return 0

    def generaSingoloVettore(self,img):
        """
        Metodo che genera il vettore di feature che descrive l'immagine passata come parametro.

        Parametri:
            img (str): path dell'immagine.
        """
        vet = self.generaSingoloVettoreAux(img,0)    # Etichetta fittizia
        return vet[:-1]     # Tolgo l'etichetta
        


    def generaSingoloVettoreAux(self, img, etichetta):
        """
        Metodo che genera il vettore di feature che descrive l'immagine passata come parametro di cui già si conosce l'etichetta. Utilizzabile per generare il traning set.
        
        Parametri:
            img (str): path dell'immagine
            etichetta (int): etichetta dell'immagine. 0 = PUBBLICITARIO, 1 = NON PUBBLICITARIO
        """
        vet=[] # Riga da scrivere nel file
        filename=self.getJsonName(img)
        print("\t File: " + filename)
        # Apro il json corrispondente alla foto analizzata
        try:
            json_file= open(filename)
        except:
            print("Errore nell'apertura del file")
            return []
        data = json.load(json_file)
        
        # Controllo pagine taggate nella didascalia
        vet.append(self.getTagDidascalia(data))

        # Controllo delle pagine taggate nella foto
        vet.append(self.getTagFoto(data,self.L))

        # Controllo se è un account verificato
        vet.append(self.getBusinessAccount(data))

        # Controllo se il post è geolocalizzato
        vet.append(self.getLocalizzato(data))

        # TODO: Da finire!
        # Controllo gli oggetti all'interno della foto
        boxes, scores, classes, nums = self.getOggetti(img,self.model)
        class_names=self.model.getClassNames()
        for iter in range(nums[0]):
            print('\t{}, {}, {}'.format(class_names[int(classes[0][iter])],
            np.array(scores[0][iter]),
            np.array(boxes[0][iter])))
        
        persona=[]
        borsa=[]
        orologio=[]
        for iter in range(nums[0]):
            print("Iter: " + str(iter) + ". Controllo persona: ")
            if len(persona)==0 and class_names[int(classes[0][iter])] == "person":
                print("Iter: " + str(iter) + ". Persona trovata: ")
                persona.append( np.array(scores[0][iter]))
                print("Score: " + str(np.array(scores[0][iter])))
                persona.append( np.array(boxes[0][iter][0] ))
                persona.append( np.array(boxes[0][iter][1] ))
                persona.append( np.array(boxes[0][iter][2] ))
                persona.append( np.array(boxes[0][iter][3] ))
            print("Iter: " + str(iter) + ". Controllo borsa: ")
            if len(borsa)==0 and class_names[int(classes[0][iter])] == "handbag":
                print("Iter: " + str(iter) + ". Borsa trovata: ")
                borsa.append(np.array(scores[0][iter]))
                print("Score: " + str(np.array(scores[0][iter])))
                borsa.append(np.array(boxes[0][iter][0] ))
                borsa.append( np.array(boxes[0][iter][1] ))
                borsa.append( np.array(boxes[0][iter][2] ))
                borsa.append( np.array(boxes[0][iter][3] ))
            print("Iter: " + str(iter) + ". Controllo orologio: ")
            if len(orologio)==0 and class_names[int(classes[0][iter])] == "clock":
                print("Iter: " + str(iter) + ". Orologio trovato: ")
                orologio.append(np.array(scores[0][iter]))
                print("Score: " + str(np.array(scores[0][iter])))
                orologio.append( np.array(boxes[0][iter][0] ))
                orologio.append( np.array(boxes[0][iter][1] ))
                orologio.append( np.array(boxes[0][iter][2] ))
                orologio.append( np.array(boxes[0][iter][3] ))
        # Li aggiungo nel vettore
        if len(persona)==0:
            persona=[0,0,0,0,0]
        if len(borsa)==0:
            borsa=[0,0,0,0,0]
        if len(orologio)==0:
            orologio=[0,0,0,0,0]
        vet+=persona
        vet+=borsa
        vet+=orologio

        # Aggiungo etichetta
        vet.append(etichetta)
        json_file.close()

        return vet

    def generaVettori(self, path_pubblicitarie, path_non_pubblicitarie, path_destinazione):
        """
        Metodo che, a partire da un insieme di immagini già classificate, genera un file contenente i corrispondenti vettori.

        Parametri:
            path_pubblicitarie (str): path della directory contenente le immagini pubblicitarie.
            path_non_pubblicitarie (str): path della directory contenente le immagini non pubblicitarie.
            path_destinazione (str): path della directory di destinazione in cui andare a scrivere il file di output.
        """
        # Apro il file csv di destinazione (dataset)
        outputFilename = 'vettori.csv'
        try:
            output_file= open(path_destinazione+outputFilename, mode='w', newline='')
        except:
            print("Errore nell'apertura del file di output.")
            exit()

        output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # Prima riga: feature names
        vet=["Pag. taggate nella didascalia", "Pag. taggate nella foto", "Business account", "Geolocalizzazione","Score persona", "X1 persona", "Y1 persona", "X2 persona", "Y2 persona","Score borsa", "X1 borsa", "Y1 borsa", "X2 borsa", "Y2 borsa","Score orologio", "X1 orologio", "Y1 orologio", "X2 orologio", "Y2 orologio",  "Pubblicitario"]
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
            files=files[0:10]

            i=0
            for img in files:
                print("Immagine " + str(i))
                vet = self.generaSingoloVettoreAux(img,j)
                if len(vet) != 0:
                    output_writer.writerow(vet)
                i=i+1
            
        output_file.close()
        print("File scritto in " + path_destinazione + outputFilename)
