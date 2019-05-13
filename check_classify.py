import numpy as np
from time import sleep
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

mnistData = []
count = 0

with open("D:\\mnist\\mnist.class") as mnistFile:	 #Путь к файлу классификации
	for mnistLine in mnistFile:
		if count % 2 == 1:
			mnistData.append([int(x) for x in mnistLine.split()])
		count += 1

resData = np.loadtxt("D:/mnist/result.txt", delimiter='\n', dtype=np.int) 	#Путь к результату классификации


print(accuracy_score(mnistData, resData, normalize=False)) 	#Точность распознавания
print(confusion_matrix(mnistData, resData, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))	#Матрица ошибок
sleep(60) 