import copy
import numpy as np

from federatedml.model_base import ModelBase
from federatedml.transfer_variable.transfer_class.hdp_vfl_transfer_variable import HdpVflTransferVariable
from federatedml.linear_model.logistic_regression.hdp_vfl.batch_data import Host
from federatedml.util import LOGGER
from federatedml.param import hdp_vfl_param
from federatedml.util import abnormal_detection
from federatedml.statistic import data_overview
from federatedml.linear_model.linear_model_weight import LRModelWeightsHost
from federatedml.util import consts
from federatedml.protobuf.generated import hdp_vfl_model_meta_pb2,hdp_vfl_model_param_pb2

class HdpVflHost(ModelBase):
    def __init__(self):
        super(HdpVflHost, self).__init__()
        self.batch_generator = Host()
        self.model_param = hdp_vfl_param.HdpVflParam()
        self.header = None
        self.model = None
        #以下三个为传输变量
        self.ir_a = None
        self.ir_b = None
        self.host_wx = None

        self.data_shape = None

    def get_header(self,data_instances):
        if self.header is not None:
            return self.header
        return data_instances.schema.get("header")

    def _init_model(self, params):
        self.epsilon = params.epsilon
        self.delta = params.delta
        self.L = params.L
        self.beta_theta = params.beta_theta
        self.beta_y = params.beta_y
        self.e = params.e
        # 这里的r指的是对于一次完整的数据集，应当经历的小批量的次数。所以每次的小批量应当等于总数据量除以r
        self.r = params.r
        self.k = params.k
        self.learning_rate = params.learning_rate
        self.lamb = params.lamb
        self.k_y = params.k_y
        # 对传输变量进行赋值
        self.transfer_variable = HdpVflTransferVariable()

    def _get_meta(self):
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
        LOGGER.info("self.data_output的值是：{}".format(self.data_output))
        for i in range(self.data_shape):
            result = "w" + str(i)
            weight_dict[result] = self.model.w[i]
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
        # 将值取出来，搞成数组的形式，然后传给self.data_output,再实际测试是否赋值给self.data_output
        self.data_output = []
        for i in range(len(result_obj.weight)):
            result = "w" + str(i)
            self.data_output.append(result_obj.weight[result])
        self.data_output = np.array(self.data_output)

    @staticmethod
    def load_data(data_instance):
        """
        设置数据标签为1和-1
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
        self.host_wx = transfer_variable.host_wx.disable_auto_clean()

    def fit(self, data_instances):
        LOGGER.info("开始纵向逻辑回归")
        #检查数据
        self._abnormal_detection(data_instances)
        #导入数据
        data_instances = data_instances.mapValues(HdpVflHost.load_data)

        # 下面开始模型的初始化
        data_shape = data_overview.get_features_shape(data_instances)
        LOGGER.info("数据的维度是:{}".format(data_shape))
        self.model = LRModelWeightsHost()
        self.model.initialize(data_shape)

        #获取数据的特征形状
        self.data_shape = data_overview.get_features_shape(data_instances)

        #批处理模块初始化
        self.batch_generator.register_batch_generator(self.transfer_variable)
        suffix = (data_instances.count(),self.r)
        self.batch_generator.initialize_batch_generator(data_instances,suffix=suffix)

        #传输变量初始化
        self.register_gradient_sync(self.transfer_variable)

        #开始正式的循环迭代的阶段
        iteration = 0 #用来记录epoches次数
        suffix_tag = 0 #用来表示传输变量表示，它的值也是最终的迭代次数
        test_suffix = ("iter",)
        while iteration <= self.e:
            for data_inst in self.batch_generator.generator_batch_data():
                LOGGER.info("----------------当前迭代次数:{}-------------------".format(suffix_tag))
                LOGGER.info("开始计算数据的内积")
                ir_b = self.model.compute_forwards(data_inst,self.model.w)

                LOGGER.info("开始生成高斯分布需要的:loc、sigma")
                loc,sigma = self.model.gaussian(self.delta,self.epsilon,self.L,self.e,
                                                int(self.r * self.e),self.learning_rate,data_inst.count(),self.k)

                LOGGER.info("开始对数据添加噪声")
                sec_ir_b = self.model.sec_intermediate_result(ir_b,loc,sigma)
                suffix_t = test_suffix + (suffix_tag,)
                LOGGER.info("当前的suffix_t值为：{}".format(suffix_t))
                LOGGER.info("开始发送给guest端sec_it_b")
                # test_transfer.send(obj=sec_ir_b,role=consts.GUEST,suffix=suffix_t)
                self.ir_b.remote(obj=sec_ir_b,role=consts.GUEST,suffix=suffix_t)

                LOGGER.info("开始从guest端接收sec_ir_a")
                sec_ir_a = self.ir_a.get(suffix=suffix_t)

                LOGGER.info("开始计算gradient_b")
                gradient_b = self.model.compute_gradient(data_inst,sec_ir_a[0],data_inst.count())

                LOGGER.info("开始更新模型参数")
                self.model.update_model(gradient_b,self.learning_rate,self.lamb)

                LOGGER.info("开始进行梯度剪切部分")
                self.model.norm_clip(self.k)

                suffix_tag += 1

            iteration += 1

        LOGGER.info("训练正式结束")
        LOGGER.info("host方的模型参数：{}".format(self.model.w))

        self.data_output = self.model.w

    def save_data(self):
        return self.data_output

    def predict(self, data_inst):
        """
        纵向逻辑回归的预测部分,由于host这里只作为参与方，所以没有输出
        Parameters
        -------------------
        data_inst:Dtable,数据的输入
        """
        LOGGER.info("------------------开始预测阶段----------------------")
        self._abnormal_detection(data_inst)
        self.data_shape = data_overview.get_features_shape(data_inst)
        #注册传输变量
        self.register_gradient_sync(self.transfer_variable)
        #预测阶段相当于重新初始化一波，所以这个时候务必注意将用到的东西重新初始化。例如最重要的weight
        #初始化模型参数
        self.model = LRModelWeightsHost()
        self.model.w = self.data_output
        data_instances = data_inst
        LOGGER.info("开始计算host方的内积wx")
        wx_host = data_instances.mapValues(lambda x : np.dot(x.features,self.data_output))
        LOGGER.info("开始将host方的内积wx发送给guest方")
        self.host_wx.remote(wx_host,role=consts.GUEST,idx=-1)
        LOGGER.info("host方完成自己的任务")
