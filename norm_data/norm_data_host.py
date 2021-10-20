from federatedml.model_base import ModelBase
from federatedml.util import LOGGER
from federatedml.param.norm_test_param import NormTestParam
from federatedml.transfer_variable.transfer_class.norm_test_transfer_variable import NormTestTransferVariable
from federatedml.util import consts
from federatedml.feature.instance import Instance

class NormTestHost(ModelBase):
    def __init__(self):
        super(NormTestHost, self).__init__()
        # 赋值参数模型
        self.model_param = NormTestParam()
        # 传输变量赋值
        self.transfer_variable = NormTestTransferVariable()

    def fit(self,data_instances):
        LOGGER.info("开始生成最大值和最小值数据表")
        LOGGER.info("查看x.features的数据类型:{}".format(type(data_instances.first()[1].features)))
        min_max_host = data_instances.mapValues(lambda x : (x.features.min(),x.features.max()))

        LOGGER.info("将min_max_host发送给guest")
        self.transfer_variable.host_to_guest.remote(obj=min_max_host,role=consts.GUEST,idx=-1,suffix=("host_to_guest",))

        LOGGER.info("接收min_max_global的值")
        min_max_global = self.transfer_variable.guest_to_host.get(idx=-1,suffix=("guest_to_host",))

        LOGGER.info("开始归一化数据")
        self.norm_data = data_instances.join(min_max_global[0],
                                             lambda x, y: Instance(features=(x.features - y[0]) / (y[1] - y[0])))

        return self.norm_data

    def save_data(self):
        return self.norm_data





