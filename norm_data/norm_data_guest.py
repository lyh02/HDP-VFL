from federatedml.model_base import ModelBase
from federatedml.util import LOGGER
from federatedml.param.norm_test_param import NormTestParam
from federatedml.transfer_variable.transfer_class.norm_test_transfer_variable import NormTestTransferVariable
from federatedml.util import consts
from federatedml.feature.instance import Instance

class NormTestGuest(ModelBase):
    def __init__(self):
        #继承基类的属性
        super(NormTestGuest, self).__init__()
        #赋值参数模型
        self.model_param = NormTestParam()
        #传输变量赋值
        self.transfer_variable = NormTestTransferVariable()
        #最终的归一化数据结果
        self.norm_data = None

    def fit(self, data_instances):
        LOGGER.info("开始生成最大值和最小值数据表")
        LOGGER.info("查看x.features的数据类型:{}".format(type(data_instances.first()[1].features)))
        min_max_guest = data_instances.mapValues(lambda x: (x.features.min(), x.features.max()))

        LOGGER.info("开始从host接收min_max_host")
        min_max_host = self.transfer_variable.host_to_guest.get(idx=-1,suffix=("host_to_guest",))

        LOGGER.info("确定真正的最大值和最小值")
        min_max_global = min_max_guest.join(min_max_host[0],lambda x,y : (min(x[0],y[0]),max(x[1],y[1])))

        LOGGER.info("发送给对方最大值和最小值")
        self.transfer_variable.guest_to_host.remote(obj=min_max_global,idx=-1,suffix=("guest_to_host",),role=consts.HOST)

        LOGGER.info("开始归一化数据")
        self.norm_data = data_instances.join(min_max_global,lambda x,y : Instance(features=(x.features-y[0]) / (y[1]-y[0]),
                                                                                  label=x.label))

        LOGGER.info("未处理之前的第一项值是：{}".format(data_instances.first()[1].features))
        LOGGER.info("处理后的第一项值是：{}".format(self.norm_data.first()[1].features))

        return self.norm_data


    def save_data(self):
        return   self.norm_data

