'''
Create maps for the comparison of policies which are learned using MDNPG, MDPGT, and PG with Entropy.
'''

import numpy as np

if __name__ == '__main__':
    map_0 = \
        [['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
         ['T', 'G', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
         ['O', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
         ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
         ['T', 'T', 'O', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
         ['T', 'T', 'T', 'T', 'T', 'O', 'T', 'T', 'T', 'O'],
         ['T', 'T', 'O', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
         ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
         ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
         ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T']]
    map_0 = np.array(map_0)

    map_1 = \
        [['T', 'T', 'T', 'T', 'O', 'T', 'T', 'T', 'T', 'T'],
         ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
         ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
         ['T', 'T', 'T', 'G', 'T', 'T', 'T', 'T', 'T', 'T'],
         ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'O'],
         ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
         ['T', 'T', 'T', 'O', 'T', 'T', 'T', 'T', 'T', 'T'],
         ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'O', 'T'],
         ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
         ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'O', 'T']]
    map_1 = np.array(map_1)

    map_2 = \
        [['T', 'T', 'T', 'T', 'T', 'O', 'T', 'T', 'T', 'T'],
         ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'O'],
         ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
         ['T', 'T', 'T', 'T', 'G', 'T', 'T', 'T', 'O', 'T'],
         ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
         ['O', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'O', 'T'],
         ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
         ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
         ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
         ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T']]
    map_2 = np.array(map_2)

    map_3 = \
        [['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
         ['T', 'T', 'T', 'T', 'T', 'T', 'O', 'T', 'O', 'T'],
         ['T', 'T', 'T', 'G', 'T', 'T', 'T', 'T', 'T', 'T'],
         ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
         ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'O', 'T'],
         ['T', 'O', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
         ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
         ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
         ['T', 'T', 'T', 'T', 'T', 'T', 'O', 'T', 'T', 'T'],
         ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T']]
    map_3 = np.array(map_3)

    map_4 = \
        [['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
         ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'O', 'T'],
         ['T', 'T', 'T', 'T', 'G', 'T', 'T', 'T', 'T', 'T'],
         ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
         ['T', 'O', 'T', 'T', 'T', 'T', 'T', 'O', 'T', 'T'],
         ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
         ['T', 'T', 'T', 'O', 'T', 'T', 'T', 'T', 'O', 'T'],
         ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
         ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
         ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T']]
    map_4 = np.array(map_4)

    np.save("map_0.npy", map_0)
    np.save("map_1.npy", map_1)
    np.save("map_2.npy", map_2)
    np.save("map_3.npy", map_3)
    np.save("map_4.npy", map_4)

    a0 = np.load("map_0.npy")
    print("0")
    print(a0)
    a1 = np.load("map_1.npy")
    print("1")
    print(a1)
    a2 = np.load("map_2.npy")
    print("2")
    print(a2)
    a3 = np.load("map_3.npy")
    print("3")
    print(a3)

    a0 = np.load("map_4.npy")
    print(a0)



