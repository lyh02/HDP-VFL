import numpy as np

from federatedml.model_base import ModelBase
from federatedml.util import LOGGER
from federatedml.secureprotol.encrypt import PaillierEncrypt
from federatedml.transfer_variable.transfer_class.hdp_vfl_transfer_variable import HdpVflTransferVariable
from federatedml.util import consts
from federatedml.linear_model.logistic_regression.hdp_vfl.batch_data import Arbiter
from federatedml.param import hdp_vfl_param

class HdpVflArbiter(ModelBase):
    def __init__(self):
        super(HdpVflArbiter, self).__init__()
        self.model_param = hdp_vfl_param.HdpVflParam()
        self.key_length = 1024
        self.transfer_variable = HdpVflTransferVariable()
        self.transfer_paillier = None
        self.cipher = PaillierEncrypt()
        self.batch_generator = Arbiter()

    def _init_model(self, params):
        self.e = params.e

    #传输变量
    def register_transfer_variable(self,transfer_variable):
        self.transfer_paillier = transfer_variable.paillier_pubkey
        self.batch_info = transfer_variable.batch_info
        #以下四个变量是为了扰动数据传输变量用的
        self.host_to_arbiter = transfer_variable.host_to_arbiter
        self.arbiter_to_host = transfer_variable.arbiter_to_host
        self.guest_to_arbiter = transfer_variable.guest_to_arbiter
        self.arbiter_to_guest = transfer_variable.arbiter_to_guest

    #正式训练过程
    def fit(self, data_instances = None,validate_data = None):
        LOGGER.info("arbiter方开始纵向逻辑回归")

        LOGGER.info("开始注册传输变量")
        self.register_transfer_variable(self.transfer_variable)

        LOGGER.info("同态加密密钥生成器开始生成")
        self.cipher.generate_key(self.key_length)

        LOGGER.info("分发公钥给其他方")
        self.transfer_paillier.remote(obj=self.cipher.get_public_key(), role=consts.HOST, idx=-1, suffix=("pub_key",))
        self.transfer_paillier.remote(obj=self.cipher.get_public_key(), role=consts.GUEST, idx=-1, suffix=("pub_key",))

        LOGGER.info("批处理过程初始化")
        batch_suffix = ("batch_info",)
        self.batch_generator.initialize_batch_generator(self.batch_info,suffix=batch_suffix)

        iteration = 0
        suffix_tag = 0
        host_to_arbiter_suffix = ("host_to_arbiter_suffix",)
        arbiter_to_host_suffix = ("arbiter_to_host_suffix",)
        guest_to_arbiter_suffix = ("guest_to_arbiter_suffix",)
        arbiter_to_guest_suffix = ("arbiter_to_guest_suffix",)
        LOGGER.info("开始迭代过程")
        while iteration <= self.e:
            for batch_index in self.batch_generator.generator_batch_data():
                #设置标签用来取数据
                suffix_ha = host_to_arbiter_suffix + (suffix_tag,)  # host方发往arbiter的加入噪音的梯度（密文）
                suffix_ah = arbiter_to_host_suffix + (suffix_tag,)  # arbiter方发往host解密后自身的梯度
                suffix_ga = guest_to_arbiter_suffix + (suffix_tag,)
                suffix_ag = arbiter_to_guest_suffix + (suffix_tag,)

                LOGGER.info("开始接收host方发来的关于guest的密文梯度")
                average_unilateral_gradient_guest_noise = self.host_to_arbiter.get(idx=-1,suffix=suffix_ha)
                LOGGER.info("开始对average_unilateral_gradient_guest_noise进行解密操作")
                gradient_guest = np.array(self.cipher.decrypt_list(average_unilateral_gradient_guest_noise[0]))
                LOGGER.info("开始将gradient_guest发送给guest方")
                self.arbiter_to_guest.remote(obj=gradient_guest,role=consts.GUEST,idx=-1,suffix=suffix_ag)

                LOGGER.info("开始接收guest方发来的关于host的密文梯度")
                average_unilateral_gradient_host_noise = self.guest_to_arbiter.get(idx=-1, suffix=suffix_ga)
                LOGGER.info("开始对average_unilateral_gradient_host_noise进行解密操作")
                gradient_host = np.array(self.cipher.decrypt_list(average_unilateral_gradient_host_noise[0]))
                LOGGER.info("开始将gradient_host发送给host方")
                self.arbiter_to_host.remote(obj=gradient_host, role=consts.HOST, idx=-1, suffix=suffix_ah)

                suffix_tag += 1

            iteration += 1
        LOGGER.info("arbiter方训练结束")


