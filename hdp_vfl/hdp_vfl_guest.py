import copy

import numpy as np

from federatedml.model_base import ModelBase
from federatedml.transfer_variable.transfer_class.hdp_vfl_transfer_variable import HdpVflTransferVariable
from federatedml.linear_model.logistic_regression.hdp_vfl.batch_data import Guest
from federatedml.util import LOGGER
from federatedml.param import hdp_vfl_param
from federatedml.util import abnormal_detection
from federatedml.linear_model.linear_model_weight import LRModelWeightsGuest
from federatedml.util import consts
from federatedml.optim import activation
from federatedml.util.io_check import assert_io_num_rows_equal
from federatedml.protobuf.generated import hdp_vfl_model_meta_pb2,hdp_vfl_model_param_pb2
from federatedml.statistic import data_overview

class HdpVflGuest(ModelBase):
    def __init__(self):
        """
        如果参数和模型参数无关，只与训练过程有关，那么就放在这里
        """
        super().__init__()
        self.batch_generator = Guest()
        self.model_param = hdp_vfl_param.HdpVflParam()
        self.transfer_variable = HdpVflTransferVariable()
        self.header = None
        #用来存最终的模型参数
        self.model = None
        #以下三个为传输变量
        self.ir_a = None
        self.ir_b = None
        self.host_wx = None

        #存取数据的维度特征
        self.data_shape = None

    def _init_model(self, params):
        """
        这里主要是将HdpVflGuest相关的参数进行赋值，所以相关的属性都放在了这里
        """
        self.epsilon = params.epsilon
        self.delta = params.delta
        self.L = params.L
        self.beta_theta = params.beta_theta
        self.beta_y = params.beta_y
        self.e = params.e
        #这里的r指的是对于一次完整的数据集，应当经历的小批量的次数。所以每次的小批量应当等于总数据量除以r
        self.r = params.r
        self.k = params.k
        self.learning_rate = params.learning_rate
        self.lamb = params.lamb
        self.k_y = params.k_y

    def _get_meta(self):
        """
        用来保存模型训练时的系列参数
        """
        meta_protobuf_obj = hdp_vfl_model_meta_pb2.HdpVflModelMeta(epsilon=self.epsilon,
                                                                   delta=self.delta,
                                                                   L=self.L,
                                                                   beta_theta=self.beta_theta,
                                                                   beta_y=self.beta_y,
                                                                   e=self.e,
                                                                   r=self.r,
                                                                   k=self.k,
                                                                   learning_rate=self.learning_rate,
                                                                   lamb=self.lamb,
                                                                   k_y=self.k_y)
        return meta_protobuf_obj

    def _get_param(self):
        """
        用来保存模型最终的训练结果信息,经过测试，运行正确
        """
        weight_dict = {}
        weight = {}
        for i in range(self.data_shape):
            result = "w" + str(i)
            weight_dict[result] = self.model.w[i]
        weight_dict["b"] = self.model.w[-1]
        weight["weight"] = weight_dict
        param_protobuf_obj = hdp_vfl_model_param_pb2.HdpVflModelParam(**weight)

        return param_protobuf_obj

    def export_model(self):
        meta_obj = self._get_meta()
        param_obj = self._get_param()
        result = {
            "HdpVflMeta": meta_obj,
            "HdpVflParam": param_obj
        }
        return result

    def load_model(self, model_dict):
        """
        这个函数是用来预测的时候才会被调用的，主要是用于将之前模型训练的相关结果拿出来，这里仅拿出来weight
        """
        result_obj = list(model_dict.get('model').values())[0].get("HdpVflParam")
        #将值取出来，搞成数组的形式，然后传给self.data_output,再实际测试是否赋值给self.data_output
        self.data_output = []
        for i in range(len(result_obj.weight)-1):
            result = "w" + str(i)
            self.data_output.append(result_obj.weight[result])
        self.data_output.append(result_obj.weight["b"])
        self.data_output = np.array(self.data_output)

    def get_header(self,data_instances):
        if self.header is not None:
            return self.header
        return data_instances.schema.get("header")

    @staticmethod
    def load_data(data_instance):
        """
        设置数据标签为1或者-1
        """
        data_instance = copy.deepcopy(data_instance)
        if data_instance.label != 1:
            data_instance.label = -1
        return data_instance

    def _abnormal_detection(self,data_instances):
        """
        主要用来检查数据的有效性
        """
        abnormal_detection.empty_table_detection(data_instances)
        abnormal_detection.empty_feature_detection(data_instances)
        ModelBase.check_schema_content(data_instances.schema)

    def register_gradient_sync(self,transfer_variable):
        self.ir_a = transfer_variable.ir_a
        self.ir_b = transfer_variable.ir_b
        self.host_wx = transfer_variable.host_wx

    def fit(self, data_instances):
        LOGGER.info("开始纵向逻辑回归")
        #检查数据
        self._abnormal_detection(data_instances)
        #导入数据
        data_instances = data_instances.mapValues(HdpVflGuest.load_data)

        #获取数据的维度特征
        self.data_shape = data_overview.get_features_shape(data_instances)

        #获取数据的header，但是这里为何是空值
        self.header = self.get_header(data_instances)

        #下面开始模型的初始化
        self.model = LRModelWeightsGuest()
        self.model.initialize(data_instances)

        #批处理数据模块初始化
        self.batch_generator.register_batch_generator(self.transfer_variable)
        #这里的self.r指的是对于一个完整的数据集，小批量的次数.这里的batch_size是一个整数
        batch_size = int(data_instances.count() / self.r )
        suffix = (data_instances.count(),self.r)
        self.batch_generator.initialize_batch_generator(data_instances,batch_size,suffix=suffix)

        # 传输变量初始化
        self.register_gradient_sync(self.transfer_variable)

        #开始正式的循环迭代训练过程,初始化迭代次数为0
        iteration = 0 #记录epoches次数
        suffix_tag = 0 #用来传输变量的标识，同时值也标识最终的传输变量的次数
        test_suffix = ("iter",)
        while iteration <= self.e:
            #获取当前批次的数据
            for data_inst in self.batch_generator.generator_batch_data():
                LOGGER.info("------------------当前迭代次数:{}-------------------".format(suffix_tag))
                LOGGER.info("从host端接收sec_ir_b")
                suffix_t = test_suffix + (suffix_tag,)
                sec_ir_b = self.ir_b.get(suffix=suffix_t)

                LOGGER.info("开始计算ir_a")
                ir_a = self.model.intermediate_result(data_inst,sec_ir_b[0],self.model.w)

                LOGGER.info("开始计算高斯噪声所需要的loc、sigma")
                loc,sigma = self.model.gaussian(self.delta,self.epsilon,self.beta_theta,self.L,
                                                self.e,int(self.e * self.r),self.learning_rate,
                                                data_inst.count(),self.k,self.beta_y,self.k_y)

                LOGGER.info("开始求sec_ir_a的值")
                sec_ir_a = self.model.sec_intermediate_result(ir_a,loc,sigma)

                LOGGER.info("开始将sec_ir_a发送给host方")
                self.ir_a.remote(obj=sec_ir_a,role=consts.HOST,idx=-1,suffix=suffix_t)

                LOGGER.info("开始计算梯度gradient_a")
                gradient_a = self.model.compute_gradient(data_inst,ir_a,data_inst.count())

                LOGGER.info("开始更新模型参数w")
                self.model.update_model(gradient_a,self.learning_rate,self.lamb)

                LOGGER.info("开始梯度剪切")
                self.model.norm_clip(self.k)

                suffix_tag += 1

            iteration += 1

        LOGGER.info("训练正式结束")
        LOGGER.info("guest方的模型参数是:{}".format(self.model.w))

        self.data_output = self.model.w

    def save_data(self):
        return self.data_output

    @assert_io_num_rows_equal
    def predict(self, data_inst):
        """
        纵向逻辑回归的预测部分
        Parameters
        -------------------
        data_inst:Dtable,数据的输入

        Returns
        -------------------
        Dtable
            输出的部分
        """
        LOGGER.info("开始预测模型的性能........")
        self._abnormal_detection(data_inst)

        #注册传输变量
        self.register_gradient_sync(self.transfer_variable)
        # 预测阶段相当于重新初始化一波，所以这个时候务必注意将用到的东西重新初始化。例如最重要的weight
        # 初始化模型参数
        self.model = LRModelWeightsGuest()
        self.model.w = self.data_output

        data_instances = data_inst
        LOGGER.info("开始计算guest方的wx内积")
        wx_guest = data_instances.mapValues(lambda x : np.dot(np.append(x.features,1),self.data_output))
        LOGGER.info("开始从host方接收host方的wx")
        wx_host = self.host_wx.get(idx=-1)
        self.data_shape = data_overview.get_features_shape(data_instances)
        #如下的wx_guest便是完整的wx
        for each_wx_host in wx_host:
            wx_guest = wx_guest.join(each_wx_host,lambda x,y : x + y)
        #将经过sigmoid函数的最终结果保存在wx_guest中
        wx_guest = wx_guest.mapValues(lambda p : activation.sigmoid(p))

        #这里可以灵活调整，这里还是设置为中间值
        threshold = 0.5
        predict_result = self.predict_score_to_output(data_instances=data_instances,predict_score=wx_guest,
                                                      classes=[0,1],threshold=threshold)
        LOGGER.info("训练结束")
        return predict_result
