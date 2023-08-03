vector = [1,0,1,1,1]

for i in range(len(vector)):
    new_vector = vector.copy()
    new_vector[i] = 1 if new_vector[i] == 0 else 0
    print(new_vector)
