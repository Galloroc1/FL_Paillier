import numpy as np
import pickle
# 实例化
# 新建公私钥
public_key,private_key = pi.createKey()
d1 = np.random.random_sample((2048,2))
print(len(pickle.dumps(d1)))

r = public_key.encrypt(d1)
print(len(pickle.dumps(r)))

r.public_key = None
print(len(pickle.dumps(r))/1024/1024)
#print(len(pickle.dumps(r.toArray())))

# def add():
#     #data1 = np.random.randint(1,5,(2, 2))
#     data1 = np.random.random_sample((2))
#     e_d1 = public_key.encrypt(data1).toArray()
#     print(e_d1)
#     e_d1 = toEncryptedNumber(e_d1)
#     print(e_d1)
#     ord = private_key.decrypt(e_d1)
#
#     # # 密文加int 加浮点
#     # e_d2 = e_d1 + 1.1 + 1
#     # print(data1+1.1+1)
#
#     #
#     # # 密文乘int 浮点
#     # e_d2 = e_d1 * 2 * 1.1
#     # print(data1* 2 * 1.1)
#     # print(private_key.decrypt(e_d2))
#
#     # # 明文int float 加密文
#     # e_d2 =1.1 + 1 + e_d1
#     # print(data1+1.1+1)
#     # print(private_key.decrypt(e_d2))
#
#     # # 密文乘int 浮点
#     # e_d2 =  2 * 1.1 * e_d1
#     # print(data1* 2 * 1.1)
#     # print(private_key.decrypt(e_d2))
#
#     # # *****************************
#     # # 密文加密文的操作中 广播机制还没有适配好
#     # # 如果两个密文的大小一样 可以直接相加
#     # # 如果是密文A = np.array([[1,1],[2,2]])
#     # #           B = [0.1]这种np原始可以广播的矩阵加法
#     # #          建议使用 [A].toArray() + [B].toArray()
#
#     # # 密文加密文 两密文矩阵相等
#     # data2 = np.random.random_sample((2, 2))
#     # e_d2 = public_key.encrypt(data2)
#     # print(data2+data1)
#     # print(private_key.decrypt(e_d1+e_d2))
#
#     # # 密文加密文 两密文矩阵不等 其中一个为np 另一个为 float
#     # data2 = 1.1
#     # e_d2 = public_key.encrypt(data2)
#     # print(data2+data1)
#     # print(private_key.decrypt(e_d1+e_d2))
#
#     # # 密文加密文 两密文矩阵不等 其中一个为np 另一个为 float
#     # data2 = np.array(1.1)
#     # #data2 = np.array([[1.1]])
#     # e_d2 = public_key.encrypt(data2)
#     # print(data1 + data2)
#     # print(private_key.decrypt(e_d1.toArray()+e_d2.toArray()))
#
# def dot():
#     data1 = np.random.random_sample((8,2))
#     data2 = np.random.random_sample((2,8))
#
#     # 密文dot明文
#     e_d1 = public_key.encrypt(data1)
#     e_d1 = e_d1.dot(data2)
#     print(np.dot(data1,data2)[0])
#     print(private_key.decrypt(e_d1)[0])
#
#
#     data1 = np.random.random_sample((3,8))
#     data2= np.random.random_sample((2,3))
#     # 明文dot密文
#     e_d1 = public_key.encrypt(data1)
#     # # 错误方式
#     # e_d1 = data2.dot(e_d1)
#     # 正确方式1 这样慢
#     # e_d1 = np.dot(data2,e_d1.toArray())
#     # 正确方式2 复杂一些 但快一些
#     e_d1 = e_d1.T.dot(data2.T).T
#     print(private_key.decrypt(e_d1)[0])
#
#     # # 密文dot密文
#     # e_d1 = public_key.encrypt(data1)
#     # e_d2 = public_key.encrypt(data2)
#     # e_d1 = e_d1.dot(e_d2)
#     # print(private_key.decrypt(e_d1)[0])
#
#
# if __name__ == '__main__':
#     add()
#     #dot()






