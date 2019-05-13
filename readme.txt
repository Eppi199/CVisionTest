Обучал из выборки mnist_png/training
Классифицировал mnist_png/testing

check_classify.py - скрипт оценки качества

mnist.train - входной файл для тренеровки
mnist.class - входной файл для классификации
result.txt - результат классификации

Библиотеки не смог загрузить из-за размера (opencv и vlveat):
https://drive.google.com/open?id=1TW7bOO40Q6zKRuuXb5YCOMeWZyDSZsgY

В архиве Release.zip находятся библиотеки, разархивировать и посместить .CVisionTest\x64\Release

Режим ввода:
  Тренировка:
		//path to image, path to save Norm, path to trainModel
  Классификация
		//path to image,  path to trainModel, path to pathNorm, result path
