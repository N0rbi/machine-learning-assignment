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

Hasonló a run_models_dim_check.py és plot_dimred.py fileok működése.

A parse_description.py valamint az explore_data.py futtatásához szükség van a data/train/train.csv filera, ezen felül a parse_description.py felhasználja a data/train/GoogleNews-vectors-negative300.txt file-t, ez beszerezhető például a [link (nagy file)](https://github.com/mmihaltz/word2vec-GoogleNews-vectors linkről).

A heatmap.py kigenerálja az adathalmazból a kovariancia mátrix vizualizációját.
