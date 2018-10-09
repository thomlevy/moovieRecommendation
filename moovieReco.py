# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 14:01:55 2018

@author: Thomas Levy
"""

import pandas as pd
import numpy as np
from surprise import Reader, Dataset, SVD

nEpochs=20 # (default=20) Change this value (e.g to 1) if you want to speed-up the learning (but with lower precision)

#Split a rating file into chunks of users
def splitChunk(inFile,outPrefix,chunkSize):
    nChunks=0
    maxUserId=0
    try:
        df= pd.read_csv(inFile, sep=",", encoding="utf8", low_memory=False)
    except Exception as e:
        print('ERROR while reading input file: '+str(e))        
        return -1
    df= pd.read_csv(inFile, sep=",", encoding="utf8", low_memory=False)
    if df.shape[0]>0:
        maxUserId=np.max(df['userId'])
    print('Maximum nb of userId: '+str(maxUserId))
    if maxUserId%chunkSize==0:
        nChunks=maxUserId // chunkSize
    else:
        nChunks=(maxUserId // chunkSize)+1
        
    #print('{0:d} chunks, size of last chunk {1:d}.'.format(nChunks, maxUserId%chunkSize ))
    
    #Create chunkId col
    df['chunkId']=df['userId'].map(lambda x: x // chunkSize)
    for i in range(nChunks):
        dfChunk=df[df['chunkId']==i]
        dfChunk.drop(['chunkId'],axis=1)
        fileName=outPrefix+'_'+str(i)+'.csv'
        dfChunk.to_csv(fileName)
        print('Chunk {0:d}, {1:d} elts stored in {2}.'.format(i,dfChunk.shape[0], fileName))
    return nChunks

#nChu=splitChunk(inFile='evaluation_ratings.csv',outPrefix='evaluation_ratings',chunkSize=13545)

#print(splitChunk(inFile='empty.csv',outPrefix='empty',chunkSize=13545))

#splitChunk(inFile='empty.csv',outPrefix='empty',chunkSize=1)

def predictRatings(ratingFile,evalFile,outFile):
    try:
        df= pd.read_csv(ratingFile, sep=",", encoding="utf8", low_memory=False)
    except Exception as e:
        print('ERROR while reading input file: '+str(e))
        return -1
    #Now learn collaborative filtering algo from input ratings
    reader = Reader()
    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    #We use Singular Value Decomposition (SVD) algo with some default parameters
    algo = SVD(n_factors=100,n_epochs=nEpochs,biased=True)
    if df.shape[0]>0:
        algo.fit(trainset)
    print('Model learned from {0}.'.format(ratingFile ))
    try:
        df= pd.read_csv(evalFile, sep=",", encoding="utf8", low_memory=False)
    except Exception as e:
        print('ERROR while reading input file: '+str(e))
        return -1    
    #def myPredict(uid,iid):
    #    return algo.predict(uid=uid,iid=iid,verbose=False).est
    #Do the prediction
    df['rating']=df['userId'].astype(object).combine(df['movieId'],lambda uid,iid: algo.predict(uid=uid,iid=iid,verbose=False).est )
    
    #Round to 3 decimal to reduce the overall memory size
    df['rating']=df['rating'].map(lambda x:round(x,3))
    df[['userId', 'movieId', 'rating']].to_csv(outFile,index=False)
    nElts=df.shape[0]
    print('{0:d} ratings predicted and stored in {1}.'.format(nElts, outFile ))
    return nElts

#print(predictRatings(ratingFile='ratings_0.csv',evalFile='evaluation_ratings_0.csv',outFile='evaluation_ratingsO.csv'))
#predictRatings(ratingFile='empty.csv',evalFile='empty.csv',outFile='emptyout.csv')

def mergeFile(inPrefix,nChunks,outFile):
    if nChunks>0:
        inFile=inPrefix+'_0.csv'
        try:
            dfAll= pd.read_csv(inFile, sep=",", encoding="utf8", low_memory=False)
        except Exception as e:
            print('ERROR while reading input file: '+str(e))
            return -1
        print('{0:d} ratings retrieved from {1}.'.format(dfAll.shape[0], inFile ))
        for i in range(1,nChunks):
            inFile=inPrefix+'_'+str(i)+'.csv'
            try:
                df= pd.read_csv(inFile, sep=",", encoding="utf8", low_memory=False)
            except Exception as e:
                print('ERROR while reading input file: '+str(e))
                return -1
            dfAll=dfAll.append(df)
            print('{0:d} ratings retrieved from {1}.'.format(df.shape[0], inFile ))
        #Round ratings to 2 decimals to reduce the overall memory size
        dfAll['rating']=dfAll['rating'].map(lambda x:round(x,2))
        dfAll[['userId', 'movieId', 'rating']].to_csv(outFile,index=False)
        nElts=dfAll.shape[0]
        print('{0:d} ratings stored in {1}.'.format(nElts, outFile ))
        return nElts
    else:
        return 0
    

#mergeFile(inPrefix='evaluation_ratings_out',nChunks=20,outFile='evaluation_ratings_out.csv')

def main():
    print("Python main function: starting")
    chunkSize=13545 #Defined to have 20 chunks (so that the last chunk has approx the same nb of users than the others)
    #Split Rating files
    nChu=splitChunk(inFile='ratings.csv',outPrefix='ratings',chunkSize=chunkSize)
    splitChunk(inFile='evaluation_ratings.csv',outPrefix='evaluation_ratings',chunkSize=chunkSize)
    #nChu=20
    for i in range(nChu):
        ratingFile='ratings_'+str(i)+'.csv'
        evalFile='evaluation_ratings_'+str(i)+'.csv'
        outFile='evaluation_ratings_out_'+str(i)+'.csv'        
        predictRatings(ratingFile,evalFile,outFile)
    
    #Now merge all eval files
    mergeFile(inPrefix='evaluation_ratings_out',nChunks=nChu,outFile='evaluation_ratings_out.csv')


if __name__ == '__main__':
    main()

