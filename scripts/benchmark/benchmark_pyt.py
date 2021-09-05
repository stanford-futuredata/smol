import time

import torch
import torchvision

def main():
    model = torchvision.models.resnet50()
    model.cuda()
    model.eval()

    NB_TRIALS = 20
    best_thpt = 0.0
    with torch.no_grad():
        for bs in range(5, 10):
            bs = 2 ** bs
            batch = torch.cuda.FloatTensor(bs, 3, 224, 224)
            # Warm up
            try:
                for i in range(10):
                    model(batch)
            except:
                break # BS too big
            begin = time.time()
            for i in range(NB_TRIALS):
                model(batch)
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
