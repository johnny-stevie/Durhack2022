import os
import sys
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense

import statistics as st
import math

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

model = Sequential()
model.add(Dense(64, input_dim=4, use_bias=True, activation='relu'))
model.add(Dense(128, use_bias=True, activation='relu'))
model.add(Dense(128, use_bias=True, activation='relu'))
model.add(Dense(64, use_bias=True, activation='relu'))
model.add(Dense(64, use_bias=True, activation='relu'))
model.add(Dense(3, use_bias=True, activation='sigmoid'))

sport = sys.argv[1]

checkpoint_path = sport + "/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model.load_weights(checkpoint_path)

#model.compile(loss='mean_squared_error',
#              optimizer='adam',
#              metrics=['mean_absolute_error'])

#model.fit(training_data, target_data, batch_size = 100, epochs=5000, verbose=2, shuffle = True, callbacks=[cp_callback])

#print(model.predict(training_data))

nationality = sys.argv[2]

goldPN = 0
goldM = 0
median = []
with open("data/" + sport + "_Gold.txt", "r") as f:
    f.readline()
    while True:
        line = f.readline()
        
        # if line is empty
        # end of file is reached
        if not line:
            break
        
        # Get next line from file
        frame = line.split(",")
        if (float(frame[1]) > 0):
            median.append(float(frame[1]))
        if (float(frame[1]) > goldM):
            goldM = float(frame[1])
        
        if (frame[0] == nationality):
            goldPN = float(frame[1])
    
median.sort()
goldM = math.sqrt(st.pvariance(median))

silverPN = 0
silverM = 0
median = [] 
with open("data/" + sport + "_Silver.txt", "r") as f:
    f.readline()
    while True:
        line = f.readline()
        
        # if line is empty
        # end of file is reached
        if not line:
            break
        
        # Get next line from file
        frame = line.split(",")
        if (float(frame[1]) > 0):
            median.append(float(frame[1]))
        if (float(frame[1]) > silverM):
            silverM = float(frame[1])
        
        if (frame[0] == nationality):
            silverPN = float(frame[1])
            break
     
        
median.sort()
silverM = math.sqrt(st.pvariance(median))

bronzePN = 0 
bronzeM = 0
median = [] 
with open("data/" + sport + "_Bronze.txt", "r") as f:
    f.readline()
    while True:
        line = f.readline()
        
        # if line is empty
        # end of file is reached
        if not line:
            break
        
        # Get next line from file
        frame = line.split(",")
        if (float(frame[1]) > 0):
            median.append(float(frame[1]))
        if (float(frame[1]) > bronzeM):
            bronzeM = float(frame[1])
        
        if (frame[0] == nationality):
            bronzePN = float(frame[1])
            break
     
        
median.sort()
bronzeM = math.sqrt(st.pvariance(median))
            
def clamp(value, min = 0, max = 0.99):
    if (value < min):
        value = min
    elif (value > max):
        value = max
    return value
            
response = ""
            
age = float(sys.argv[3])
weight = float(sys.argv[4])
height = float(sys.argv[5])
sex = float(sys.argv[6])
            
data = model.predict([[19,50,160,0]])[0]
response += "baseline: chance of gold (" + str(round(data[0] * 100, 2)) + "%), silver (" + str(round(data[1] * 100, 2)) + "%), bronze (" + str(round(data[2] * 100, 2)) + "%)\n"
            
gold = data[0]
if (goldPN >= goldM):
    gold = gold * (1 + goldPN - goldM)
    
silver = data[1]
if (silverPN >= silverM):
    silver = silver * (1 + silverPN - silverM)
    
bronze = data[1]
if (bronzePN >= bronzeM):
    bronze = bronze * (1 + bronzePN - bronzeM)
            
finalized = [clamp(gold), clamp(silver), clamp(bronze)]
response += "nationality influence: chance of gold (" + str(round(finalized[0] * 100, 2)) + "%), silver (" + str(round(finalized[1] * 100, 2)) + "%), bronze (" + str(round(finalized[2] * 100, 2)) + "%)\n"

if (len(sys.argv) == 13):
    d = 1
    with open("data/" + sport + "_data.txt", "r") as f:
        f.readline()
        data = f.readline()
        frame = data.split(",")
        
        d = float(frame[0])

    goldB = float(sys.argv[7])
    goldN = float(sys.argv[8])
    silverB = float(sys.argv[9])
    silverN = float(sys.argv[10])
    bronzeB = float(sys.argv[11])
    bronzeN = float(sys.argv[12])

    finalized = [(d * finalized[0] + (goldB - 0.5) * math.pow(goldN, 1/4)) / d, (d * finalized[0] + (silverB - 0.5) * math.pow(silverN, 1/4)) / d, (d * finalized[0] + (bronzeB - 0.5) * math.pow(bronzeN, 1/4)) / d]
    response += "sentimental analysis: chance of gold (" + str(round(finalized[0] * 100, 2)) + "%), silver (" + str(round(finalized[1] * 100, 2)) + "%), bronze (" + str(round(finalized[2] * 100, 2)) + "%)\n"

with open("results.txt", "w") as f:
    f.write(response)