#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


import numpy as np

from federatedml.framework.weights import ListWeights, TransferableWeights
from federatedml.secureprotol.fate_paillier import PaillierEncryptedNumber
from federatedml.util import LOGGER
from federatedml.statistic import data_overview


class LinearModelWeights(ListWeights):
    def __init__(self, l, fit_intercept, raise_overflow_error=True):
        l = np.array(l)
        if not isinstance(l[0], PaillierEncryptedNumber):
            if np.max(np.abs(l)) > 1e8:
                if raise_overflow_error:
                    raise RuntimeError("The model weights are overflow, please check if the "
                                       "input data has been normalized")
                else:
                    LOGGER.warning(f"LinearModelWeights contains entry greater than 1e8.")
        super().__init__(l)
        self.fit_intercept = fit_intercept
        self.raise_overflow_error = raise_overflow_error

    def for_remote(self):
        return TransferableWeights(self._weights, self.__class__, self.fit_intercept)

    @property
    def coef_(self):
        if self.fit_intercept:
            return np.array(self._weights[:-1])
        return np.array(self._weights)

    @property
    def intercept_(self):
        if self.fit_intercept:
            return self._weights[-1]
        return 0.0

    def binary_op(self, other: 'LinearModelWeights', func, inplace):
        if inplace:
            for k, v in enumerate(self._weights):
                self._weights[k] = func(self._weights[k], other._weights[k])
            return self
        else:
            _w = []
            for k, v in enumerate(self._weights):
                _w.append(func(self._weights[k], other._weights[k]))
            return LinearModelWeights(_w, self.fit_intercept)

class LRModelWeightsGuest():
    def __init__(self,w = None):
        self.w = w
        #偏置放在guest这一边
        #self.intercept = intercept

    def initialize(self,data_instances):
        """
        对w和偏置b进行初始化
        """
        data_shape = data_overview.get_features_shape(data_instances)
        LOGGER.info("除去偏置b,数据的维度属性是：{}".format(data_shape))
        # 将偏置b也加进来
        if isinstance(data_shape, int):
            data_shape += 1
        # 初始化模型参数
        self.w = np.random.rand(data_shape)
        LOGGER.info("guest方初始化模型参数self.w是：{}".format(self.w))

    def compute_gradient(self,data_instances,ir_a,b):
        """
        计算梯度
        parameters:
        ---------------
        data_instances:Dtable
        ir_a:中间结果
        b:当前的小批量数据多少个
        """
        result_tables = data_instances.join(ir_a,lambda x,y : np.append(x.features,1) * y)
        result = 0
        for result_table in result_tables.collect():
            result += result_table[1]

        #result就是最终的结果，类型应该是numpy数组
        gradient_a = result / b
        return gradient_a

    def update_model(self,gradient_a,eta,lamb):
        """
        更新模型参数
        parameters
        ---------------
        gradient_b:梯度
        eta:学习率
        lamb:正则化参数
        """
        LOGGER.info("self.w的值：{}".format(self.w))
        LOGGER.info("gradient_a的值：{}".format(gradient_a))
        self.w -= eta * (gradient_a + lamb * self.w)

    def gaussian(self,delta,epsilon,beta_theta,L,e,T,eta,b,k,beta_y,k_y,length):
        """
        生成高斯噪声所需要的loc、sigma_b
        parameters
        ---------------------------
        delta:非严格的DP损失
        epsilon:隐私保护预算
        beta_theta:smooth parameters,未知
        L:lipschitz 常数，默认值为1
        e:epochs
        T:e * r,表示总的迭代次数
        eta:learning rate
        b:mini-batch size
        k:梯度剪切参数
        beta_y:smooth parameters,未知
        k_y:target bound,未知
        length:host方数据的维度，作为全局敏感度的一个参数
        """
        loc = 0
        partial_1 = np.sqrt(2 * np.log(1.25 / delta))

        partial_2_1 = 4 * np.square(beta_theta) * np.square(L) * np.square(e) * T * np.square(eta) / b
        partial_2_2 = 8 * (beta_theta * k + beta_y * k_y) * beta_theta * L * np.square(e) * eta / b
        partial_2_3 = 4 * np.square(beta_theta * k + beta_y * k_y) * e
        partial_2 = np.sqrt(partial_2_1 + partial_2_2 + partial_2_3) * np.sqrt(length) / b #最大敏感度

        partial_3 = epsilon

        sigma_b = partial_1 * partial_2 / partial_3
        LOGGER.info("在guest方，全局敏感度值:{}".format(partial_2))
        LOGGER.info("在guest方，标准差sigma_b:{}".format(sigma_b))
        return loc,sigma_b

    def sec_intermediate_result(self,average_unilateral_gradient_host,loc,sigma_b):
        """
        parameters
        ir_a:是一个Dtable格式数据表，值为标量
        ----------------------------
        return
        在ir_a的基础上添加噪声
        -----------------------
        算法
        同Host方，不再赘述
        """
        length = average_unilateral_gradient_host.shape[0]
        sec_result_0 = []
        for i in range(length):
            sec_result_0.append(np.random.normal(loc, sigma_b))
        LOGGER.info("添加的噪音为：{}".format(sec_result_0))
        average_unilateral_gradient_guest_noise = average_unilateral_gradient_host + sec_result_0

        return average_unilateral_gradient_guest_noise


        # #第一种方法
        # sec_result_1 = []
        # for ir_a_tuple_1 in ir_a.collect():
        #     test_tuple_1 = (ir_a_tuple_1[0], np.random.normal(loc, sigma_b))
        #     sec_result_1.append(test_tuple_1)
        #
        # #第二种方法
        # noise_gaussion = float(np.random.normal(loc, sigma_b) / ir_a.count())
        # sec_result_2 = []
        # for ir_a_tuple_1 in ir_a.collect():
        #     test_tuple_1 = (ir_a_tuple_1[0], noise_gaussion)
        #     sec_result_2.append(test_tuple_1)
        #
        # #第三种方式
        # sec_result_3 = []
        # for ir_a_tuple_1 in ir_a.collect():
        #     test_tuple_1 = (ir_a_tuple_1[0], float(np.random.normal(loc, sigma_b) / ir_a.count()))
        #     sec_result_3.append(test_tuple_1)
        #
        # #第二种方法
        # # gaussian_noise = np.random.normal(loc, sigma_b, ir_a.count())
        # # gaussian_noise.tolist()
        # # sec_result_2 = []
        # # first_data_id = ir_a.first()[0]
        # # for ir_b_tuple_2 in ir_a.collect():
        # #     test_tuple_2 = (ir_b_tuple_2[0], gaussian_noise[int(ir_b_tuple_2[0]) - int(first_data_id)])
        # #     sec_result_2.append(test_tuple_2)
        #
        # # -----------------------------------------------------------------
        # # 将高斯噪声封装成Dtbale格式，切勿重新初始化一个session。直接用之前的就可以了
        # # computing_session.init(work_mode=0, backend=0, session_id="gaussian id")
        # gaussian_noise = computing_session.parallelize(sec_result_1, partition=4, include_key=True)
        # # 扰动数据内积
        # sec_result = ir_a.join(gaussian_noise, lambda x, y: x + y)
        # LOGGER.info("原始数据是：{}".format(list(ir_a.collect())))
        # LOGGER.info("高斯噪声是:{}".format(list(gaussian_noise.collect())))
        # LOGGER.info("扰动后，数据值是：{}".format(list(sec_result.collect())))
        # return sec_result

    def intermediate_result(self,data_instances,sec_ir_b,w):
        """
        计算IR_A,这里要注意数据的维度必须扩展一个，因为还是偏置b
        parameters
        -------------------
        data_instances:A方的数据表
        sec_ir_b:B方添加过噪声的数据内积
        w:A方的模型参数
        return
        -------------------
        一个Dtable格式的数据，里面存储的是梯度的一部分
        """
        ir_a = data_instances.join(sec_ir_b,lambda x ,y : \
            ( 1 / (1 + np.exp(-x.label * (np.dot(np.append(x.features,1),w) + y))) - 1) * x.label)
        return ir_a

    def norm_clip(self,k):
        """
        parameters
        ------------
        k:这个大小自己来定义，此函数主要用来防止梯度爆炸
        """
        result = np.sqrt(np.sum(np.square(self.w))) / k
        if result > 1:
            self.w /= result


class LRModelWeightsHost():
    def __init__(self,w = None):
        #host方没有偏置，偏置b都放在了guest方
        self.w = w

    def initialize(self, data_shape):
        """
        对w进行初始化
        """
        #这里的w格式是numpy数组
        self.w = np.random.rand(data_shape)
        LOGGER.info("Host初始化模型参数self.w是：{}".format(self.w))

    def compute_gradient(self,data_instances,ir_a,b):
        result_tables = data_instances.join(ir_a, lambda x, y: x.features * y)
        result = 0
        for result_table in result_tables.collect():
            result += result_table[1]

        # result就是最终的结果，类型应该是numpy数组
        gradient_b = result / b
        return gradient_b

    def compute_forwards(self, data_instances, w):
        """
        计算wx内积
        parameters:
        ---------------
        data_instances:Dtable
        w:numpy数组
        return:
        ----------------
        Dtable，值为wx
        """
        ir_b = data_instances.mapValues(lambda x : np.dot(x.features,w))
        return ir_b

    def sec_intermediate_result(self,average_unilateral_gradient_guest,loc,sigma_a):
        """
        备注：此函数已经测试过了，正常运行
        添加噪声
        算法
        ------------------
        1.根据当前批次的数据取出对应的ID
        2.拼接ID和高斯噪声
        3.调用API接口中的join()函数进行数据扰动
        """
        #这里的sec_result是一个列表，里面存取的都是元组，元组的第一项都是当前批次数据的ID，第二项便是高斯噪声
        #这里的疑问点在于是否可以
        length = average_unilateral_gradient_guest.shape[0]
        sec_result_0 = []
        for i in range(length):
            sec_result_0.append(np.random.normal(loc,sigma_a))
        LOGGER.info("添加的噪音为：{}".format(sec_result_0))
        average_unilateral_gradient_guest_noise = average_unilateral_gradient_guest + sec_result_0

        return average_unilateral_gradient_guest_noise


        #以下若干种方法暂时略去
        # #第一种添加噪声的方式，噪声会很大
        # sec_result_1 = []
        # for ir_b_tuple_1 in ir_b.collect():
        #     test_tuple_1 = (ir_b_tuple_1[0],np.random.normal(loc,sigma_a))
        #     sec_result_1.append(test_tuple_1)
        #
        # # 第二种方法，噪声均分
        # noise_gaussion = float(np.random.normal(loc, sigma_a) / ir_b.count())
        # sec_result_2 = []
        # for ir_a_tuple_1 in ir_b.collect():
        #     test_tuple_1 = (ir_a_tuple_1[0], noise_gaussion)
        #     sec_result_2.append(test_tuple_1)
        #
        # #第三种方式，噪声随机均分
        # sec_result_3 = []
        # for ir_a_tuple_1 in ir_b.collect():
        #     test_tuple_1 = (ir_a_tuple_1[0], float(np.random.normal(loc, sigma_a) / ir_b.count()))
        #     sec_result_3.append(test_tuple_1)
        # # #第二种添加噪声的方式
        # # gaussian_noise = np.random.normal(loc, sigma_a, ir_b.count())
        # # gaussian_noise.tolist()
        # # sec_result_2 = []
        # # first_data_id = ir_b.first()[0]
        # # for ir_b_tuple_2 in ir_b.collect():
        # #     test_tuple_2 = (ir_b_tuple_2[0],gaussian_noise[int(ir_b_tuple_2[0]) - int(first_data_id)])
        # #     sec_result_2.append(test_tuple_2)
        #
        # #-----------------------------------------------------------------
        # #将高斯噪声封装成Dtbale格式
        # # computing_session.init(work_mode=0,backend=0,session_id="gaussian id")
        # gaussian_noise = computing_session.parallelize(sec_result_1,partition=4,include_key=True)
        # #扰动数据内积
        # LOGGER.info("原始数据是:{}".format(list(ir_b.collect())))
        # LOGGER.info("高斯噪声为：{}".format(list(gaussian_noise.collect())))
        # sec_result = ir_b.join(gaussian_noise,lambda x,y : x + y)
        # LOGGER.info("添加完噪声后：{}".format(list(sec_result.collect())))
        # return sec_result

    def gaussian(self,delta,epsilon,L,e,T,eta,b,k,length):
        """
        生成高斯噪声所需要的loc、sigma_a
        parameters
        --------------------
        delta:一定程度的允许错误的值，因为高斯机制非严格满足DP机制
        epsion:隐私保护预算
        L:lipschitz 常数，默认值为1
        e:epochs
        T:e * r,表示总的迭代次数
        eta:learning rate
        b:mini-batch size
        k:梯度剪切参数
        length:表述数据维度的长度

        loc,sigma都是高斯分布的俩参数
        """
        loc = 0
        partial_1 = np.sqrt(2 * np.log(1.25 / delta))
        partial_2_1 = (4 * np.square(L) * np.square(e) * T * np.square(eta)) / b
        partial_2_2 = (8 * k * L * np.square(e) * eta) / b
        partial_2_3 = 4 * np.square(k) * e
        partial_2 = np.sqrt(partial_2_1 + partial_2_2 + partial_2_3) * np.sqrt(length) / b #最大敏感度
        partial_3 = epsilon
        sigma_a = partial_1 * partial_2 / partial_3

        LOGGER.info("站在host方，全局敏感度值:{}".format(partial_2))
        LOGGER.info("站在host方，标准差是:{}".format(sigma_a))
        return loc,sigma_a

    def norm_clip(self, k):
        """
        parameters
        ------------
        k:这个大小自己来定义，此函数主要用来防止梯度爆炸
        """
        result = np.sqrt(np.sum(np.square(self.w))) / k
        if result > 1:
            self.w /= result

    def update_model(self, gradient_b, eta, lamb):
        """
        更新模型参数
        parameters
        ---------------
        gradient_b:梯度
        eta:学习率
        lamb:正则化参数
        """
        self.w -= eta * (gradient_b + lamb * self.w)



