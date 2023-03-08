import os
import cv2
import pandas as pd
from deepface import DeepFace


#img = cv2.imread('faces/f.jpeg')                                                                               # Read image
#results = DeepFace.analyze(img, actions=("gender", "age", "race", "emotion"))                                  # Analyze image
#print(results)                                                                                                 # Print results


# create a dictionary data for our images
data = {                                                                                                        # Create a dictionary data for our images 
    "Name": [],                                                                                                     
    "Age": [],
    "Gender": [],
    "Race": []
}

# Iterate through the images file, load the images & extract information from them, & create a dataset for them
for file in os.listdir("faces"):                                                                                                                
    result = DeepFace.analyze(cv2.imread(f"faces/{file}"), actions=("gender", "age", "race"))                                                   # Analyze image                                                                                       
    data["Name"].append(file.split(".")[0])                                                                                                     # Append name to data                
    data["Age"].append(result[0]["age"])                                                                                                        # Append age to data            
    data["Gender"].append(result[0]["dominant_gender"])                                                                                         # Append gender to data                                                                                                        
    data["Race"].append(result[0]["dominant_race"])                                                                                             # Append race to data    


# convert data
df = pd.DataFrame(data)                                                                                                                         # Convert data to dataframe  
print(df)                                                                                                                                       # Print dataframe                  

df.to_csv("people.csv")                                                                                                                         # Save dataframe to csv file     

