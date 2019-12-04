import os

from nltk.corpus import wordnet

# path= "C:/Users/kiran_000/Desktop/MS/Data Mining/Actual Project/Images/2k_images/"
# files= os.listdir(path)
#
# for i,file in enumerate(files):
#     os.rename(os.path.join(path, file), os.path.join(path, ''.join([str(i), '.jpg'])))

for syn in wordnet.synsets("best"):
    for l in syn.lemmas():
        string.append(str(l.name()))