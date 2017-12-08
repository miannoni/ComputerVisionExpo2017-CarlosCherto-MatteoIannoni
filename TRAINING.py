from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from random import randint
from math import *
from scipy import stats as st
from mnist import *
import pickle

train_images = read_idx('train-images.idx3-ubyte')
train_labels = read_idx('train-labels.idx1-ubyte')

def cluster_dict():
	equivalencia = {}

	for cluster in range(0,clustquant):
		lista_reais = []
		for index in range(0,len(lista)):
			if kmeans.labels_[index] == cluster:
				lista_reais.append(train_labels[lista[index]])
			
		equivalencia[str(cluster)] = st.mode(lista_reais, axis=None)[0][0]

	return (equivalencia)

################################################################################################################################################
################################################################################################################################################
#														Multi cluster classifier trainer
################################################################################################################################################
################################################################################################################################################

# for c in range(0,10):

	x = 0

	breakit = False

	index_lim = 10000

	lista = []
	X = []

	while len(lista) < index_lim:

		if (x not in lista) and (train_labels[x] == c) and (len(lista) <= index_lim*0.95):
			lista.append(x)
			X.append(np.concatenate((train_images[x])))

		if (x not in lista) and (train_labels[x] != c) and (len(lista) >= index_lim*0.95):
			lista.append(x)
			X.append(np.concatenate((train_images[x])))

		x += 1

		if x == 59999:
			x = 0
			if breakit == True:
				break
			else:
				breakit = True

		if x%500 == 0:
			print(x)

	X = np.array(X)

	clustquant = 25
	kmeans = KMeans(n_clusters=clustquant, random_state=0).fit(X)
	
	equivalencia = cluster_dict()

	save = [kmeans,equivalencia]

	pickle.dump( save, open( 'specific_train_' + str(clustquant) + 'clust_' + str(index_lim) + 'imgs_number_' + str(c) + '.p', "wb" ) )
	print(c)

################################################################################################################################################
################################################################################################################################################



################################################################################################################################################
################################################################################################################################################
#															Roots classifier trainer
################################################################################################################################################
################################################################################################################################################


cluster_per_number = 15
image_quantity = 30000

def train_classifier_1(clusters, image_number):

	numbers = {}

	clustquant = clusters

	limit = image_number

	for number in range(0,9):

		numbers[number] = []

		x = 0

		breaker = False

		while (x < 50000) and (len(numbers[number]) < (image_number/10)):
			if (train_labels[x] == number):
				numbers[number].append(np.concatenate((train_images[x])))

			if x >= 50000:
				x = 0
				if breaker == True:
					break
				if breaker == False:
					breaker = True

			x += 1

		kmeans = KMeans(n_clusters=clustquant, random_state=0).fit(np.array(numbers[number]))

		numbers[number] = kmeans.cluster_centers_

	pickle.dump(numbers, open('class_1_train_' + str(clusters) + 'clust_' + str(image_number) + 'imgs.p', "wb" ) )

train_classifier_1(cluster_per_number,image_quantity)

################################################################################################################################################
################################################################################################################################################




















# print(len(np.split(kmeans.cluster_centers_[0],28)))

# i = 4

# print(kmeans.labels_[i])
# print(equivalencia[str(i)])

####################################################################################################################################################################
####################################################################################################################################################################
####################################################################################################################################################################
####################################################################################################################################################################



# def chute(imagem):
# 	distance = None
# 	guess = None

# 	for key in numbers:
# 		for cluster in range(0,len(numbers[key])):

# 			# calc_dist = sqrt(np.sum(np.subtract(numbers[key][cluster],np.concatenate((imagem)))**2))
# 			# print(calc_dist)
# 			# print(np.linalg.norm(numbers[key][cluster]-np.concatenate((imagem))))
# 			calc_dist = np.linalg.norm(numbers[key][cluster]-np.concatenate((imagem)))

# 			if distance == None:
# 				distance = calc_dist
# 				guess = key
			
# 			if distance > calc_dist:
# 				distance = calc_dist
# 				guess = key

# 	return guess

# def runstats():

# 	acertos = 0
# 	erros = 0

# 	for y in range(1000):

# 		x = randint(40000,50000)

# 		img = train_images[x]
# 		label = train_labels[x]

# 		guess = chute(img)

# 		if guess == label:
# 			acertos += 1
# 		else:
# 			erros += 1

# 	print((acertos/(acertos + erros))*100)

# def visual_test():

# 	x = randint(40000,50000)

# 	img = train_images[x]

# 	print('Guess: ' + str(chute(img)))
# 	print('Actual Value: ' + str(train_labels[x]))

# 	plt.imshow(img)
# 	plt.show()

# cluster_per_number = 20
# image_quantity = 10000

# # train_classifier_1(cluster_per_number,image_quantity)
# numbers = pickle.load(open('class_1_train_' + str(cluster_per_number) + 'clust_' + str(image_quantity) + 'imgs.p', "rb"))

# # runstats()

# ########################################

# # counter = 0

# # while (counter < 1000):
# # 	x = randint(0,50000)

# # 	print(chute(train_images[x]))
# # 	print(train_labels[x])

# # 	counter += 1

# ########################################

# visual_test()