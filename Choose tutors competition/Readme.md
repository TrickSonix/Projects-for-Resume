Задача из соревнования https://www.kaggle.com/c/choose-tutors
===============
Ваша задача этом соревновании - предсказать вероятность того, подойдет ли репетитор для подготовки к экзамену по математике. Вам будут даны два датасета: train.csv (содержит признаки и целевую переменную) и test.csv (только признаки).
Ограничения на импорт библиотек:
```
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
```
---------
Метрика: ROC_AUC
---------
Датасет состоит из довольно чистых данных с различными признаками о репетиторах и целевой переменной - бинарной классификации подходит репетитор или нет.
---------
Задача решалась с помощью Логистической регрессии и Классификатора основанного на Случайном Лесе.
---------
Лучшее значение метрики показала Логистическая регрессия. </br>
Train score: 0.775 </br>
Test score: 0.73394
---------
