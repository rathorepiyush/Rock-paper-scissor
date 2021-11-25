from keras.models import Sequential
from keras.layers import Dense,MaxPool2D,Flatten
from keras.optimizers import Adam
from keras.applications.densenet import DenseNet121
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
# from skimage import io
import cv2
import random

def prepImg(pth):
    return cv2.resize(pth,(300,300)).reshape(1,300,300,3)

def updateScore(play,bplay,p,b):
    winRule = {'rock':'scissor','scissor':'paper','paper':'rock'}
    if play == bplay:
        return p,b
    elif bplay == winRule[play]:
        return p+1,b
    else:
        return p,b+1
    
    # serialize model to JSON

"""DenseNet"""

densenet = DenseNet121(include_top=False, weights='imagenet', classes=3,input_shape=(300,300,3))
densenet.trainable=True
def genericModel(base):
    model = Sequential()
    model.add(base)
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense(3,activation='softmax'))
    model.compile(optimizer=Adam(),loss='categorical_crossentropy',metrics=['acc'])
    return model
dnet = genericModel(densenet)
model_json = dnet.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
dnet.save_weights("modelweight.h5")
print("Saved model to disk")

# with open('model.json', 'r') as f:
#     loaded_model_json = f.read()
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("modelweight.h5")
print("Loaded model from disk")

shape_to_label = {'rock':np.array([1.,0.,0.]),'paper':np.array([0.,1.,0.]),'scissor':np.array([0.,0.,1.])}
arr_to_shape = {np.argmax(shape_to_label[x]):x for x in shape_to_label.keys()}

options = ['rock','paper','scissor']
winRule = {'rock':'scissor','scissor':'paper','paper':'rock'}
rounds = 0
botScore = 0
playerScore = 0

cap = cv2.VideoCapture(1)
ret,frame = cap.read()
loaded_model.predict(prepImg(frame[50:350,100:400]))

NUM_ROUNDS = 3
bplay = ""
while True:
    ret , frame = cap.read()
    frame = cv2.putText(frame,"Press Space to start",(160,200),cv2.FONT_HERSHEY_SIMPLEX,1,(250,250,0),2,cv2.LINE_AA)
    cv2.imshow('Rock Paper Scissor',frame)
    if cv2.waitKey(1) & 0xff == ord(' '):
        break

for rounds in range(NUM_ROUNDS):
    pred = ""
    for i in range(90):
        ret,frame = cap.read()
    
        # Countdown    
        if i//20 < 3 :
            frame = cv2.putText(frame,str(i//20+1),(320,100),cv2.FONT_HERSHEY_SIMPLEX,3,(250,250,0),2,cv2.LINE_AA)

        # Prediction
        elif i/20 < 3.5:
            pred = arr_to_shape[np.argmax(loaded_model.predict(prepImg(frame[50:350,100:400])))]
        
        # Get Bots Move
        elif i/20 == 3.5:
            bplay = random.choice(options)            
            print(pred,bplay)

        # Update Score
        elif i//20 == 4:
            playerScore,botScore = updateScore(pred,bplay,playerScore,botScore)
            break

        cv2.rectangle(frame, (100, 150), (300, 350), (255, 255, 255), 2)
        frame = cv2.putText(frame,"Player : {}      Bot : {}".format(playerScore,botScore),(120,400),cv2.FONT_HERSHEY_SIMPLEX,1,(250,250,0),2,cv2.LINE_AA)
        frame = cv2.putText(frame,pred,(150,140),cv2.FONT_HERSHEY_SIMPLEX,1,(250,250,0),2,cv2.LINE_AA)
        frame = cv2.putText(frame,"Bot Played : {}".format(bplay),(300,140),cv2.FONT_HERSHEY_SIMPLEX,1,(250,250,0),2,cv2.LINE_AA)        
        cv2.imshow('Rock Paper Scissor',frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

if playerScore > botScore:
    winner = "You Won!!"
elif playerScore == botScore:
    winner = "Its a Tie"
else:
    winner = "Bot Won.."
    
while True:
    ret,frame = cap.read()
    frame = cv2.putText(frame,winner,(230,150),cv2.FONT_HERSHEY_SIMPLEX,1,(250,250,0),2,cv2.LINE_AA)
    frame = cv2.putText(frame,"Press q to quit",(190,200),cv2.FONT_HERSHEY_SIMPLEX,1,(250,250,0),2,cv2.LINE_AA)
    frame = cv2.putText(frame,"Player : {}      Bot : {}".format(playerScore,botScore),(120,400),cv2.FONT_HERSHEY_SIMPLEX,1,(250,250,0),2,cv2.LINE_AA)
    cv2.imshow('Rock Paper Scissor',frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
