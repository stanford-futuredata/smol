import time

import keras
import numpy as np

def main():
    model = keras.applications.resnet50.ResNet50(weights=None)

    NB_TRIALS = 20
    best_thpt = 0.0
    for bs in range(5, 10):
        bs = 2 ** bs
        batch = np.random.rand(bs, 224, 224, 3)
        try:
            for i in range(10):
                model.predict(batch)
        except:
            break
        begin = time.time()
        for i in range(NB_TRIALS):
            model.predict(batch)
        end = time.time()
        thpt = bs * NB_TRIALS / (end - begin)
        if thpt > best_thpt:
            best_thpt = thpt
        print('Benchmarked', bs)
        print('Throughput is', thpt)

        time.sleep(10) # so the T4 doesn't catch on fire
    print(best_thpt)

if __name__ == '__main__':
    main()
