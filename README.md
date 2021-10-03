# 论文复现完成
复现论文：《Hybrid Differentially Private Federated Learning on Vertically Partitioned Data》.主要是复现到fate当中
备注：这个项目是在fate框架上才可以运行，所以这里只给出核心代码。需要完整的配置文件的，请私聊我哦~（经过测试，可以在fate框架中正常跑通）。

norm_data文件夹下主要用于处理数据的L2标准化（这个部分非必要部分，只是为了处理数据）
hdp_vfl为主文件夹
  batch_data.py主要用来生成批处理数据，这里的难点在于如何让双方同步的取出相同ID的数据。这里的代码逻辑参考fate框架中已有的纵向算法代码中对批处理数据的处理
  linear_model_weight.py为算法的核心部分，包括高斯噪声的生成、数据的扰动等等
  hdp_vfl_guest.py文件为guest方主调用模块
  hdp_vfl_host.py文件为host方主调用模块
  
  
备注：
待优化的部分：测试集、训练集，还有如何优化论文算法本身。后续会慢慢添加进来~
