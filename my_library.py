import matplotlib.pyplot as plt
from scipy import stats as st
from random import randint
import numpy as np
import math
import cv2

##################################################################################################################################################################################################
##################################################################################################################################################################################################
##################################################################################################################################################################################################
##################################################################################################################################################################################################

def simpleton (screen,x,y,w,h):

	# Make image square

	if w < h:
		diferenca = int(round((h - w)/2))

		x -= diferenca
		w += 2*diferenca

	elif w > h:
		diferenca = int(round((w - h)/2))

		y -= diferenca
		h += 2*diferenca

	# adicionar borda

	borda = int(round(w*0.15))

	w += 2*borda
	x -= borda
	h += 2*borda
	y -= borda

	# Downscale img

	imagem = screen[y:y+h,x:x+w]

	new_img = []

	if str(type(imagem)) != 'NoneType':

		length = len(imagem)/28

		while len(new_img) < 28:
			line = []
			while len(line) < 28:
				line.append(round(np.mean(imagem[round(length*len(new_img)):round(length*(len(new_img) + 1)),round(length*len(line)):round(length*(len(line) + 1))])))
			new_img.append(line)

		return	new_img

	else: return img















##################################################################################################################################################################################################
##################################################################################################################################################################################################
##################################################################################################################################################################################################
##################################################################################################################################################################################################

def find_center(img):
	x, y = np.indices(img.shape)
	concat = np.concatenate(img)
	total = np.sum(concat)

	x = x.ravel(order='F')
	y = y.ravel(order='F')

	x = np.multiply(x,concat)
	y = np.multiply(y,concat)

	x = np.sum(x)
	y = np.sum(y)

	x = x/total
	y = y/total

	return [x,y]




def sig(x, deriv=False):
	if deriv:
		return np.multiply(sig(x),(1-sig(x)))
	return 1/(1+np.exp(-x))

def feedforward(i,w,b):

	z = []
	a = []
	a.append(i)
	cont=0
	while cont<2:
		dp = np.dot(w[cont],a[cont])
		z.append(np.add(dp,b[cont]))
		var = sig(z[cont])
		a.append(var)
		cont+=1
	return a,z







##################################################################################################################################################################################################
##################################################################################################################################################################################################
##################################################################################################################################################################################################
##################################################################################################################################################################################################




def COM (mass):

	position = np.arange(1,np.size(mass) + 1,1)

	top = np.sum(np.multiply(position,mass))
	bottom = np.sum(mass)

	return (top/bottom)

def center_x_y (img):

	if (len(img) > 0) and (len(img[0]) > 0):
		lines = round(COM(np.array([np.mean(img[x]) for x in range(0,len(img))])))
		columns = round(COM(np.array([np.mean(img[:,x]) for x in range(0,len(img[0]))])))

		return [lines, columns]

def translate (img,lista):

	ymin,ymax,xmin,xmax = lista

	line_center_pixel, column_center_pixel = center_x_y(img)

	print(line_center_pixel)

	value1 = (line_center_pixel - ((ymax-ymin)/2))
	value2 = (line_center_pixel + ((ymax-ymin)/2))
	

	if (value1 < 0):
		value1 = 0
		value2 = (ymax-ymin)
	
	if (value2 > len(img)):
		value1 = len(img) - ymax + ymin - 1
		value2 = len(img) - 1

	value3 = (column_center_pixel - ((xmax-xmin)/2))
	value4 = (column_center_pixel + ((xmax-xmin)/2))

	if (value3 < 0):
		value3 = 0
		value4 = (xmax-xmin)

	if (value4 > len(img)):
		value3 = len(img) - xmax + xmin - 1
		value4 = len(img) - 1

	ymin = value1
	ymax = value2
	xmin = value3
	xmax = value4

	return [ymin,ymax,xmin,xmax]

def center_img (img):
	img = np.array(img)

	if (len(img) > 0):
		if (len(img[0]) > 0):

			size = len(img)

			length1 = len(img)
			length2 = len(img[0])

			line_center_pixel, column_center_pixel = center_x_y(img)

			if not (math.isnan(line_center_pixel)) and not (math.isnan(column_center_pixel)):

				if (line_center_pixel <= size):
					size -= (size - line_center_pixel)

				if ((line_center_pixel + size) >= len(img)):
					size -= ((line_center_pixel + size) - len(img))

				if (column_center_pixel <= size):
					size -= (size - column_center_pixel)

				if ((column_center_pixel + size) >= len(img[0])):
					size -= ((column_center_pixel + size) - len(img[0]))

				if ((line_center_pixel - size) >= 0) and ((line_center_pixel + size) <= len(img)) and ((column_center_pixel - size) >= 0) and ((column_center_pixel + size) <= len(img[0])):
					ymin = int(line_center_pixel - size)
					ymax = int(line_center_pixel + size)
					xmin = int(column_center_pixel - size)
					xmax = int(column_center_pixel + size)

				return [ymin,ymax,xmin,xmax]#translate(img,[ymin,ymax,xmin,xmax])
	else: return img

def find_number(img):

	if (type(img) != 'NoneType'):

		imagem = img

		first_flag = False

		counter = 0

		limit = 10

		# mean = np.mean(imagem)

		lista = translate(imagem,[0,280,0,280])

		imagem = imagem[lista[0]:lista[1],lista[2]:lista[3]]



#########################################################################################################################################################
		# centers = center_x_y(imagem)
		# compare = centers

		# while (np.mean(imagem) < limit) and (np.mean(imagem) > 0):
		# 	imagem = imagem[2:-2,2:-2]
		# 	compare = center_x_y(imagem)

		# 	if centers != compare:
		# 		imagem = center_img(imagem)
		# 		centers = compare
#########################################################################################################################################################

		new_img = []

		if str(type(imagem)) != 'NoneType':

			length = len(imagem)/28

			while len(new_img) < 28:
				line = []
				while len(line) < 28:
					line.append(round(np.mean(imagem[round(length*len(new_img)):round(length*(len(new_img) + 1)),round(length*len(line)):round(length*(len(line) + 1))])))
				new_img.append(line)

			return	new_img

		else: return img
	else: return img

def cluster_dict(cluster_quantity, image_quantity, kmeans, train_labels):
	equivalencia = {}

	for cluster in range(0,cluster_quantity):
		lista_reais = []
		for index in range(0,image_quantity):
			if kmeans.labels_[index] == cluster:
				lista_reais.append(train_labels[index])
		
		equivalencia[str(cluster)] = st.mode(lista_reais, axis=None)[0][0]

	return (equivalencia)

def run_stats_classifier_0(equivalencia,image_quantity,kmeans,train_labels,train_images):
	
	statistics = {
					'Correct' : 0,
					'Incorrect' : 0
	}

	tries = 1000

	for n in range(0,tries):
		
		pred = randint(image_quantity,len(train_images)-1)

		if equivalencia[str(kmeans.predict([np.concatenate((train_images[pred]))])[0])] == train_labels[pred]:
			statistics['Correct'] += 1
		else:
			statistics['Incorrect'] += 1

	print('% de acertos: ' + str(statistics['Correct']*100/tries))
	print('% de erros: ' + str(statistics['Incorrect']*100/tries))