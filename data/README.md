# Dataset

The code expects `data/Crop_recommendation.csv`. The file is not committed: it comes from
Kaggle and is redistributed here only by reference, so you have to fetch it yourself.

## Where to get it

Canonical source: the Crop Recommendation Dataset on Kaggle,
https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

Download `Crop_recommendation.csv` from that page and put it in this directory. If you have
the Kaggle CLI configured:

    kaggle datasets download -d atharvaingle/crop-recommendation-dataset -p data --unzip

`scripts/download_data.py` does the same job from a public mirror of the same file and checks
the result against the expected schema. It is a convenience, not the authoritative source.

## Expected shape

2200 rows, 8 columns, no missing values. 22 crop labels with exactly 100 rows each.

| column      | type  | meaning                          |
|-------------|-------|----------------------------------|
| N           | int   | nitrogen content in soil (kg/ha) |
| P           | int   | phosphorus content in soil       |
| K           | int   | potassium content in soil        |
| temperature | float | degrees Celsius                  |
| humidity    | float | relative humidity, percent       |
| ph          | float | soil pH                          |
| rainfall    | float | millimetres                      |
| label       | str   | crop name, the prediction target |

First row of the file, for reference:

    90,42,43,20.87974371,82.00274423,6.502985292000001,202.9355362,rice

The labels are: apple, banana, blackgram, chickpea, coconut, coffee, cotton, grapes, jute,
kidneybeans, lentil, maize, mango, mothbeans, mungbean, muskmelon, orange, papaya,
pigeonpeas, pomegranate, rice, watermelon.

`crop_recommendation.data.load_dataset` validates the columns, the dtypes and the absence of
missing values on load, so a file with the wrong shape fails immediately rather than halfway
through training.
