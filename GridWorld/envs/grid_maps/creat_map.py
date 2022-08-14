import numpy as np

if __name__ == '__main__':
    map_init = \
        [['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
         ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
         ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
         ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
         ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
         ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
         ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
         ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
         ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
         ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T']]
    np.save("map_init.npy", map_init)
    map = np.load("map_init.npy")
    print(map)

