Collaborators: Dave, Xander, Tao, and Karim

Project Proposal:
We propose to develop a machine learning model that learns technical documents to provide a summary of input documents of the same field. 

The steps to the project:
Find a technical corpus
Clean corpus
Vectorize corpus
Choose an ML Model (RNN, K-mean, Seq to Seq)
Test model and record model stats.
Design a portal to demonstrate model stats.


structure

    ├── data                    
    │   ├── inputs          # Risk Documents to be processesed
    │   └── outputs         # Model, Vectorizer, CSV Files
    │        ├── colortexts   #HTML color coded docs 
    │        └── summaryCsv  # Makup of each doc, by clustter       
    ├── app                       
    │   ├── predict.py          
    │   └── predict.py             
    ├── notebooks                
    │   ├── Glove.ipynb          
    │   └── kmeans_pipeline.ipynb               
    ├── LICENSE      
    └── README.md


Libraries Used
collections
colorama
colorsys
datetime
future
gensim
heapq
jinja2
math
matplotlib
markdown
markdown2
nltk
numpy
operator
pandas
Path
re
scipy
seaborn
sklearn
string
sys
time
tqdm
