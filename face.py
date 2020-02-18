import os
import cv2
import numpy as np
from PIL import Image
import pickle

base_dir=os.path.dirname(os.path.abspath(__file__))
image_dir=os.path.join('images',base_dir)

face_cascade = cv2.CascadeClassifier('road_sign.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id=0
label_img={}
x_label=[]
x_train=[]

for root,dir,files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path=os.path.join(root,file)
            label=os.path.basename(root)
            if not label in label_img:
                label_img[label]=current_id
                current_id+=1
            id_img=label_img[label]
            print(id_img);
            pil_image=Image.open(path).convert("L")
            final_img=pil_image.resize((550,550),Image.ANTIALIAS)
            image_array=np.array(final_img , "uint8")

            face=face_cascade.detectMultiScale(image_array,scaleFactor=1.3,minNeighbors=5)

            for i in face:
                (x,y,w,h)=i
                rh=image_array[y:y+h,x:x+w]
                x_label.append(id_img)
                x_train.append(rh)

print(x_label)                
                
with open('label_data','wb') as f:
    pickle.dump(label_img,f)

face_recognizer.train(x_train,np.array(x_label))
face_recognizer.save('saved_reco.yml')
