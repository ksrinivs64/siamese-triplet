from textdatasets import ANNBasedTripletSelector, TripletDataset

"""
   This test tests to see if 
"""
selector = ANNBasedTripletSelector('../fuzzyjoiner/names_to_cleanse/peoplesNames.txt')
triples_train = TripletDataset(selector, True)
triples_test = TripletDataset(selector, True)
print(len(triples_train))
print(len(triples_test))

print(triples_train[0][0][0].size())