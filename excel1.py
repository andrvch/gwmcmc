print("Files from table 1 which aren't in table 2:")
x = [(f,t,s) for f,t,s in fts1 if (f,t) not in ft2]
print(x)
