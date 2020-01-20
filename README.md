# MLJazzAtiam

**Project:** Predict chord progressions of jazz music, represented as coherent chord label sequences with the help of probabilistic models.

**Domain:** Machine Learning.

**Contributors:** Zitian Gao, Théo Golvet, Colette Voisembert, Yujia Yang.

**Mentor:** Tristan Carsault.

**Description:** This project aims to predict chord progressions of jazz music, represented as coherent chord label sequences with the help of probabilistic models. In this study, we propose to use different neural network models to generate symbolic chord sequences. Besides, we study the impact of the introduction of different musical distances through the loss function during the training of our models. Thus, we want to improve existing methods by doing multi-step prediction and by injecting music theory knowledge through the learning method in order to be able to perform accurate prediction of chord sequence and jazz melody generation. Ultimately, this project could be used to perform automatic accompaniment and improvisation.

## GitHub

### Prerequisites
You need:
         <ul>
         <li> **Python version:**  Python 3.7.1 (2020-01-20) </li>
         <li> Pytorch </li>
         <li>Libraries needed : numpy, os, random, time, math, torch, matplotlib.pyplot
         </ul>

### Installation
You must:
         <ul>
         <li>Download the GitHub repository</li>
         <li>Download the [dataset](https://github.com/keunwoochoi/lstm_real_book) and put it in the data foder.</li>
         </ul>
         
### Launch the application
You need to:
         <ul>
        <li>Remove 'READMD.txt' in the 'jazz_xlab' folder.
        <li>**Preprocessing:** <ul>
                             <li>Create "train", "test" and "validation" folders in [data](https://github.com/ColetteVois/MLJazzAtiam/tree/master/data).
                             <li>Change the "../data/.." in "data/" or change the place of your data folder.
                             <li>Run [split_dataset.py](https://github.com/ColetteVois/MLJazzAtiam/blob/master/split_dataset.py).
                             <li>Run [preProcessing.py](https://github.com/ColetteVois/MLJazzAtiam/blob/master/preProcessing.py).
                             </ul>   
            **OR** use the already made files .csv in the repository and go directly to the next step.
        <li>Run the [main.py](https://github.com/ColetteVois/MLJazzAtiam/blob/master/main.py).file</li>
        </ul>

  
### Small description of the repository
Description:
        <ul>
        <li>préProcessing/dataloader: [preProcessing.py](https://github.com/ColetteVois/MLJazzAtiam/blob/master/preProcessing.py), [createdataloader.py](https://github.com/ColetteVois/MLJazzAtiam/blob/master/create_dataloader.py), [splitdataset.py](https://github.com/ColetteVois/MLJazzAtiam/blob/master/split_dataset.py), [get_the_list.py](https://github.com/ColetteVois/MLJazzAtiam/blob/master/get_the_list.py), [create_dataloader.py](https://github.com/ColetteVois/MLJazzAtiam/blob/master/chordUtil.py, https://github.com/ColetteVois/MLJazzAtiam/blob/master/create_dataloader.py) Zitian, Théo            
        <li>alphabet: [alphabet_exacte.py](https://github.com/ColetteVois/MLJazzAtiam/blob/master/alphabet_exacte.py), [chordUtil.py](https://github.com/ColetteVois/MLJazzAtiam/blob/master/alphabet_redu.py), [chordUtil.py](https://github.com/ColetteVois/MLJazzAtiam/blob/master/chordUtil.py), [create_dataloader.py](https://github.com/ColetteVois/MLJazzAtiam/blob/master/create_dataloader.py) Yujia, Colette
        <li>training: [ChordsAlphabets.py](https://github.com/ColetteVois/MLJazzAtiam/blob/master/ChordsAlphabets.py), [ChordsAlphabets.py](https://github.com/ColetteVois/MLJazzAtiam/blob/master/ChordsAlphabets.py) Théo                 
        <li>evaluation: [ChordsAlphabets.py](https://github.com/ColetteVois/MLJazzAtiam/blob/master/ChordsAlphabets.py), [ChordsAlphabets.py](https://github.com/ColetteVois/MLJazzAtiam/blob/master/ChordsAlphabets.py) Théo                
        <li>ACE: [ACEScore.py](https://github.com/ColetteVois/MLJazzAtiam/blob/master/ACEScore.py), [ChordsAlphabets.py](https://github.com/ColetteVois/MLJazzAtiam/blob/master/ChordsAlphabets.py), [ChordsAlphabets.py](https://github.com/ColetteVois/MLJazzAtiam/blob/master/ChordsAlphabets.py), [distance.py](https://github.com/ColetteVois/MLJazzAtiam/blob/master/distance.py) Zitian, Yujia
        <li>loss function: [ChordsAlphabets.py](https://github.com/ColetteVois/MLJazzAtiam/blob/master/ChordsAlphabets.py), [representation.py](https://github.com/ColetteVois/MLJazzAtiam/blob/master/representation.py) Zitian, Colette
         </li>
        </ul>
