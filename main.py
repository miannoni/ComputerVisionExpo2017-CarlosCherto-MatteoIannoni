import matplotlib.pyplot as plt
from random import randint
from my_library import *
from scipy import stats
from mnist import *
import scipy.misc
import pickle
import cv2

train_images = read_idx('train-images.idx3-ubyte')
train_labels = read_idx('train-labels.idx1-ubyte')

#################################################################
#			classifier 0 setup	(cluster, basico)				#
#################################################################

image_quantity = 59999
cluster_quantity = 100

training_0 = pickle.load(open('train_' + str(cluster_quantity) + 'clust_' + str(image_quantity) + 'imgs.p', "rb"))

equivalence_dictionary = cluster_dict(cluster_quantity, image_quantity, training_0, train_labels)

run_stats_classifier_0(equivalence_dictionary,image_quantity,training_0,train_labels,train_images)

def classificador0(kick):
	return((equivalence_dictionary[str(training_0.predict([np.concatenate(kick)])[0])]))

#################################################################

#################################################################
#				classifier 1 setup (handmade, raiz)				#
#################################################################

# image_quantity = 30000
# cluster_quantity = 15

# classifier1 = pickle.load(open('class_1_train_' + str(cluster_quantity) + 'clust_' + str(image_quantity) + 'imgs.p', "rb"))

# def classificador1(kick):

	# distance = None
	# guess = None

	# for key in classifier1:
	# 	for cluster in range(0,len(classifier1[0])):

	# 		calc_dist = np.linalg.norm(classifier1[key][cluster]-np.concatenate((kick)))

	# 		if distance == None:
	# 			distance = calc_dist
	# 			guess = key
			
	# 		if distance > calc_dist:
	# 			distance = calc_dist
	# 			guess = key

	# return guess

#################################################################

#################################################################
#				classifier 2 setup (multicluster)				#
#################################################################

# image_quantity = 5000
# cluster_quantity = 25

# for x in range(0,10):
# 	exec("training_" + str(x) + ",dictionary" + str(x) + " = pickle.load(open('specific_train_" + str(cluster_quantity) + "clust_" + str(image_quantity) + "imgs_number_" + str(x) + ".p', 'rb'))")

# def classificador3(kick):
	# guess0 = (dictionary0[str(training_0.predict([np.concatenate(kick)])[0])])
	# guess1 = (dictionary1[str(training_1.predict([np.concatenate(kick)])[0])])
	# guess2 = (dictionary2[str(training_2.predict([np.concatenate(kick)])[0])])
	# guess3 = (dictionary3[str(training_3.predict([np.concatenate(kick)])[0])])
	# guess4 = (dictionary4[str(training_4.predict([np.concatenate(kick)])[0])])
	# guess5 = (dictionary5[str(training_5.predict([np.concatenate(kick)])[0])])
	# guess6 = (dictionary6[str(training_6.predict([np.concatenate(kick)])[0])])
	# guess7 = (dictionary7[str(training_7.predict([np.concatenate(kick)])[0])])
	# guess8 = (dictionary8[str(training_8.predict([np.concatenate(kick)])[0])])
	# guess9 = (dictionary9[str(training_9.predict([np.concatenate(kick)])[0])])

	# return (guess0,guess1,guess2,guess3,guess4,guess5,guess6,guess7,guess8,guess9)

#################################################################


#################################################################
#				classifier 3 setup (rede neural)				#
#################################################################

w_neural,b_neural = pickle.load(open('neural.p', "rb"))

def classificador3(kick):
	# kick = np.array(kick)

	a,z = feedforward(kick,w_neural,b_neural)

	cont2=0
	m=max(a[2])

	while cont2<10:
		if m==a[2][cont2]:
			break
		cont2+=1

	return (cont2)

#################################################################




cap = cv2.VideoCapture(0)

counter = 0

mode = []

while(1):

	_, frame = cap.read()

	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	hsv = cv2.medianBlur(hsv,3)

	lower_blue = np.array([90, 70, 30])	#<<<<<<<<<<<<<<<< ESSES VALORES FUNCIONAM NA SALA
	upper_blue = np.array([160, 255, 245])

	mask = cv2.inRange(hsv, lower_blue, upper_blue)

	mask = cv2.medianBlur(mask,3)


	th = cv2.adaptiveThreshold(mask,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,21,5)
	im2, contours, hierarchy = cv2.findContours(th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	
	# try:
	if (np.mean(mask) > 0) and (len(contours) > 1):
		# try:
			x,y,w,h = cv2.boundingRect(contours[1])
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
			cv2.putText(frame, 'LOCKED-ON TARGET',(x+w+10,y+h),0,0.3,(0,255,0))

			cv2.drawContours(frame, contours[1], -1, (0,255,0), 3)

			#find_number(mask[x-200:x+w+200,y-100:y+h+100])

			finder = simpleton(mask,x,y,w,h)

			plt.imshow(finder)
			plt.show()

			print(classificador3(np.resize(finder,(784,1))))

			finder = cv2.resize(np.array(finder),(640,640))

		# except ValueError:
		# 	finder = np.array(np.zeros((640,480)))
	else:
		finder = np.array(np.zeros((640,480)))
	# except TypeError:
	# 	finder = np.array(np.zeros((640,480)))
	# except IndexError:
	# 	finder = np.array(np.zeros((640,480)))

	# im = Image.fromarray(th)
	# im.save("img_1.jpg")

	scipy.misc.imsave('img_1.jpg', cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
	scipy.misc.imsave('img_2.jpg', mask)
	scipy.misc.imsave('img_3.jpg', th)
	scipy.misc.imsave('img_4.jpg', finder)

	cv2.imshow('thres',th)
	cv2.imshow('mask', mask)
	cv2.imshow('Frame',frame)
	cv2.imshow('FindNumber',finder)

	# print(mode)
	counter += 1
	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		break

cv2.destroyAllWindows()