# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 12:17:19 2018

@author: Thomas Levy
"""

import unittest

import moovieReco

class UnitTestMethods(unittest.TestCase):

    #Basic unit testing of all the methods from moovieReco module withn an empty csv file (i.e with no ratings)
    
    def test_splitEmpty(self):
        self.assertEqual(moovieReco.splitChunk(inFile='empty.csv',outPrefix='empty',chunkSize=1), 0)
    
    def test_predictEmpty(self):
        self.assertEqual(moovieReco.predictRatings(ratingFile='empty.csv',evalFile='empty.csv',outFile='emptyout.csv'),0)
    
    def test_mergeEmpty(self):
        self.assertEqual(moovieReco.mergeFile(inPrefix='empty',nChunks=0,outFile='out.csv'),0)        

if __name__ == '__main__':
    unittest.main()

