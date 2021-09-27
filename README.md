# 论文复现中.....
复现论文：《Hybrid Differentially Private Federated Learning on Vertically Partitioned Data》.主要是复现到fate当中
备注：这个项目是在fate框架上才可以运行，所以这里只给出核心代码（经过测试，可以在fate框架中正常跑通）。

norm_data文件夹下主要用于处理数据的L2标准化
hdp_vfl为主文件夹
  batch_data.py主要用来生成批处理数据，这里的难点在于如何让双方同步的取出相同ID的数据。这里的代码逻辑参考fate框架中已有的纵向算法代码中对批处理数据的处理
