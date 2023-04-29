import pickle
f = open('./datasets/semeval14/laptop_test.raw.graph', 'rb')

a = pickle.load(f)
# b = list(a.keys())# .sort()
print(a[1026])
