import cv2

#our image
img_file='b.jpg'

#our pre trained-classifier
classifier_file='cars.xml'

#create opencv image
img=cv2.imread('car2.jpg')

#convert the grayscale
black_n_white=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#create car classifier
car_tracker=cv2.CascadeClassifier(classifier_file)

#detect cars
cars=car_tracker.detectMultiScale(black_n_white)
#print(cars)

#draw rectangles around the cars
#car1=cars[2]
#print((car1))
#(x,y,w,h)=car1
for (x,y,w,h) in cars:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255),2)

#display the image
cv2.imshow('Mohammad', img)

#dont autoclose
cv2.waitKey()

print('code completed')