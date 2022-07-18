# aerial-image-matching
The final solution of the problem of "Цифровой Прорыв" competition about the positioning of the image on the ground.

## Основные результаты
По метрике конкурса использование только подложки дает метрику соревнования **0.92** тогда как использование тренировочного набора, который более свежий по времени дает **0.99**. Тренировочный и тестовый набор снят в одно время и алгоритм находит совпадения даже в структуре облаков! Что, кажется, немного портит идею конкурса.

## Идея решения
### Deep learning (спойлер: не вошел в финальное решение)
Вначале я пробовал решать задачу методами deep learning, мне очень понравился подход из [этой статьи](https://arxiv.org/pdf/1703.05593.pdf) об использовании корреляции пирсона для фич из претрейн-моделей, который по духу аналогичен классическому подходу сопоставления снимков. Блокнот с тренировкой модели в [блокноте](https://github.com/ifserge/aerial-image-matching/blob/main/cnn_geom_model.ipynb).  Вывод который я сделал после его использования - он **очень** чувствителен к изменению масштаба и сильно упростить себе работу с ним не удается. Плюсом он классно справляется, если мы закрываем все слоем облаков, такой набор аугментаций очень похож на тренировочный датасет:
```python
from imgaug import augmenters as iaa
seq = iaa.Sequential([
    iaa.AddToBrightness((-10,10)),
    iaa.Clouds(),
])
```

### Классический подход
Совсем классический подход на SIFT не дал хороших результатов, но использование __**афинно инвариантного A-SIFT**__ из [статьи](https://www.ipol.im/pub/art/2011/my-asift/) и оптимизированного для GPU кода из [репозитория](https://github.com/Mars-Rover-Localization/PyASIFT/blob/main/asift.py) полностью решили проблему поворотов и вопроса дейтсвительно ли это одинаковый кадр. Дополнил этот подход Flann, 2NN по дескрипторам с обычным условием, что топ-1 гораздо ближе топ-2. Для борьбы с недостатком ключевых точек в условиях облачности использовался адаптивный контраст CLAHE. _Само решение_ - KNN поиск ближайшего соседа по матрике отношения количества сопоставленных ключевых точек к общему их числу. В конце концов алгоритм оказался настолько точен, что хватило K=1
![пример матча облачных картинок с CLAHE](https://github.com/ifserge/aerial-image-matching/blob/main/clouds.png)

Отдельно хочу отметить алгоритм для сопоставления ключевых [AdaLAM](https://github.com/cavalli1234/AdaLAM) оптимизированный для многопоточности, но что намного важнее у него отличные настраиваемые фильтры, которые сходу отсекают очень много случайных совпадений ключевых точек. Единственная проблема, для картинок больше 256x256 надо очень много памяти.

## Структура репозитория
- [showcase.ipynb](https://github.com/ifserge/aerial-image-matching/blob/main/showcase.ipynb) **демонстрация финального** подхода с объяснениями по шагам
- [cnn_geometric_model.py](https://github.com/ifserge/aerial-image-matching/blob/main/cnn_geometric_model.py) pytorch-модули для работы с корреляциями
- [cnn_geom_model.ipynb](https://github.com/ifserge/aerial-image-matching/blob/main/cnn_geom_model.ipynb)  очень сырой ноутбук с тренировкой модели, которая учится предсказывать координаты патча и поворот 1024x1024 на патче размером 3072х3072, в итоговое решение не вошел
- [asift.py](https://github.com/ifserge/aerial-image-matching/blob/main/asift.py) цельнотянут из репозитория a-sift, выражаю благодарность авторам
- [zero_stage.py](https://github.com/ifserge/aerial-image-matching/blob/main/zero_stage.py) скрипт для генерации заготовки решения (список файлов и т.д.)
- [first_stage.py](https://github.com/ifserge/aerial-image-matching/blob/main/first_stage.py) первая стадия решения: генерим соответствия, если у нас много кейпойнтов
- [second_stage.py](https://github.com/ifserge/aerial-image-matching/blob/main/second_stage.py) вторая стадия: если не хватает кейпойнтов, делаем CLAHE, картинки сматченные на первой стадии в последующие не идут
- [third_stage.py](https://github.com/ifserge/aerial-image-matching/blob/main/third_stage.py) третья стадия: для картинок, которые не сматчились ни на первой, ни на второй стадии делаем еще более жесткую контрастность и сравниваем с расширенным набором картинок из тренировочного набора с добавлением CLAHE и без него
- [aerial_sub.py](https://github.com/ifserge/aerial-image-matching/blob/main/aerial_sub.py) генерация сабмита для конкурса
- [requirements.txt](https://github.com/ifserge/aerial-image-matching/blob/main/requirements.txt) python-пакеты необходимы для запуска решения

## Запуск решения
### структура папок
```
unzip -qq train_dataset_train.zip &&\
mkdir test &&\
unzip -qq -d ./test test_dataset_test.zip
```
в папке ./sub/ будут находится итоговые json-файлы

### подготовка решения
```
pip install -r requirements.txt
```

### запуск решения
```
python zero_stage.py &&\
python first_stage.py &&\
python second_stage.py &&\
python third_stage.py &&\
python third_stage.py &&\
python third_stage.py &&\
python aerial_sub.py
```
Третья стадия вызывается несколько раз в силу того, что одна из фотографий тестового набора иногда не матчится с первого раза, но с третьего точно :)

## Послесловие
Очень понравилась задача! Немного нестандартная, хотя и оказалась довольно простенькой по итогу. Кажется, что если бы тренировочный набор не дали или дали бы десяток картинок из него, решение было бы полезнее, но уж вышло как вышло. С матчингом фотографий и кейпойнтами и гомологией никогда раньше не работал. На DL-подход я потратил часов 12, на классический подход еще 8 часов времени. Использовал виртуалку из YCloud прерываемую с A100 за 104руб./час.
