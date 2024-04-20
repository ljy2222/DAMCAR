from torch import nn
from torch.autograd import Function

from layers.core import DNN
from model.base_model_new import BaseModel
from preprocessing.inputs import combined_dnn_input


class GRL(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class ADA(BaseModel):
    def __init__(self, linear_feature_columns, dnn_feature_columns, use_fm=True, dnn_hidden_units=(256, 128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001,
                 seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False, task='binary',
                 device='cpu', gpus=None):
        super(ADA, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                     device=device, gpus=gpus)
        self.encoder = DNN(self.compute_input_dim(dnn_feature_columns), (300, 200, 100),
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                           use_bn=dnn_use_bn, init_std=init_std, device=device)
        self.source_encoder = DNN(100, (128, 100),
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                           use_bn=dnn_use_bn, init_std=init_std, device=device)
        self.label_generator = nn.Linear(100, 1, bias=False).to(device)
        self.domain_classifier = nn.Linear(100, 1, bias=False).to(device)
        self.GRL = GRL()

    def forward(self, inputs, source):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(inputs, self.dnn_feature_columns, self.embedding_dict)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
        dnn_output = self.encoder(dnn_input)
        if source:
            dnn_output = self.source_encoder(dnn_output)
        label_logit = self.label_generator(dnn_output)
        label_pred = self.out(label_logit)
        dnn_output = GRL.apply(dnn_output, 1)
        domain_logit = self.domain_classifier(dnn_output)
        domain_pred = self.out(domain_logit)
        return label_pred, domain_pred