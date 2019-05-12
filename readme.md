A fájlok futtatásához létre kell hozni egy data mappát, amiben az adathalmaz található meg.
A modellek a nevüknek megfelelő fájlokban érhetőek el külön-külön.
    ann2.py = 3 rétegű neuron háló
    decisiontree.py = döntési fa
    gmm.py = Gaussian Mixture Model
    knn.py = K Nearest neighbors
    randomforest.py = Random Forest Classifier

Ezek külön-külön futtathatóak, ez miatt mindegyikben megtalálható a jellemző feldolgozás és dimenziócsökkentés is.
A run_models.py összehúzza a fent említett modelleket, és 4 különböző random_seed-en futtatja őket, majd csv-t generál belőle (out.csv).
Az így kapott csv-t aztán a plot.py futtatásával lehet pdf-re konvertálni.
tpot.py tartalmazza a TPOT csomag használatát. Ennek futtatása után létrejön a pipeline.py, ami tartalmazni fogja a TPOT szerinti legjobb modelt a legjobb paraméterezéssel.
