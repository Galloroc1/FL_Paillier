import datetime
import random
import numpy as np
import sys

encrypt = True
decrypt = True
dot = True
is_pool = False

def new(data):
    from paillier.paillier import generate_paillier_keypair
    p,q = generate_paillier_keypair()
    if encrypt:
        st = datetime.datetime.now()
        encry = p.encrypt(data, is_pool=is_pool, partitions=10)
        print("encrypt",datetime.datetime.now() - st)

    if decrypt:
        st = datetime.datetime.now()
        decry = q.decrypt(encry, is_pool=is_pool, partitions=10)
        print("decrypt",datetime.datetime.now() - st)

    if dot:
        st = datetime.datetime.now()
        d2 = np.random.random_sample((100,2))
        encry = encry.dot(d2, is_pool=is_pool, partitions=10)
        # encry = p.encrypt(data, is_pool=False)
        print("dot",datetime.datetime.now() - st)
        # print(data.dot(d2))
        # print(q.decrypt(encry))

def old(data):
    from np_paillier.paillier import generate_paillier_keypair
    p, q = generate_paillier_keypair()
    if encrypt:
        st = datetime.datetime.now()
        encry = p.encrypt(data, is_pool=is_pool)
        print("encrypt", datetime.datetime.now() - st)

    if decrypt:
        st = datetime.datetime.now()
        decry = q.decrypt(encry, is_pool=is_pool)
        print("decrypt", datetime.datetime.now() - st)

    if dot:
        st = datetime.datetime.now()
        encry = encry.dot(np.random.random_sample((100,2)), is_pool=is_pool)
        # encry = p.encrypt(data, is_pool=False)
        print("dot", datetime.datetime.now() - st)



# 0:00:04.252671
if __name__ == '__main__':
    from my_paillier.paillier import generate_paillier_keypair
    data = np.random.random_sample((100, 100))
    new(data)
    old(data)


# 1000 * 1000   pool
# new                       old
# encrypt 0:00:02.378278    encrypt 0:00:01.811769
# decrypt 0:01:19.849919    decrypt 0:01:16.421048


# 1000000 * 1   pool
# new                       old
# encrypt 0:00:02.298825    encrypt kill
# decrypt 0:01:13.777810    decrypt kill


# 100 * 100 dot 100 * 100   pool
# new                       old
# dot     0:00:26.407451    dot     0:00:16.841429

# 100000 * 2 dot 2 * 2   pool
# new                       old
# dot     0:00:09.384771    dot     0:00:19.200635

# 100 * 100   pool   dot(100,2)
# new                       old
# encrypt 0:00:00.567879    encrypt 0:00:00.568909
# decrypt 0:00:01.147182    decrypt 0:00:01.121322
# dot     0:00:01.595924    dot 0:00:01.112450


# 100 * 100   not pool   dot(100,2)
# new                       old
# encrypt 0:00:00.044421    encrypt 0:00:00.069014
# decrypt 0:00:03.895211    decrypt 0:00:03.855885
# dot     0:00:02.936037    dot 0:00:02.670889

