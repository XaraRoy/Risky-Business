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
    ├── app                       
    │   ├── model_generator.py           
    │   ├── string_predict.py        
    │   └── folder_predict.py             
                   
    │   ├── Glove.ipynb          
    │   └── kmeans_pipeline.ipynb         
    │    folder_predict.py        
    ├── LICENSE      
    └── README.md


Libraries Used
collections
datetime
future
gensim
heapq
math
matplotlib
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