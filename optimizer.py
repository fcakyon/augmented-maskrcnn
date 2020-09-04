from torch_optimizer import AdaBound, Lamb, RAdam
from torch.optim import SGD, Adam


class OptimizerFactory:
    def __init__(
        self,
        learning_rate,
        momentum=0,
        weight_decay=0,
        betas=[0.9, 0.999],
        eps=1e-08,
        amsgrad=False,
        adabound_gamma=1e-3,
        adabound_final_lr=0.1,
    ):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps
        self.amsgrad = amsgrad
        self.adabound_gamma = adabound_gamma
        self.adabound_final_lr = adabound_final_lr

    def get(self, params, optimizer_name):
        """
        Creates torch optimizer specified by 'optimizer_name' for given 'params'.

        params: list of torch.nn.parameter.Parameter
        optimizer_name: str
        """
        if optimizer_name == "sgd":
            optimizer = SGD(
                params,
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        elif optimizer_name == "adam":
            optimizer = Adam(
                params,
                lr=self.learning_rate,
                betas=tuple(self.betas),
                eps=self.eps,
                weight_decay=self.weight_decay,
                amsgrad=self.amsgrad,
            )
        elif optimizer_name == "adabound":
            optimizer = AdaBound(
                params,
                lr=self.learning_rate,
                betas=tuple(self.betas),
                final_lr=self.adabound_final_lr,
                gamma=self.adabound_gamma,
                eps=self.eps,
                weight_decay=self.weight_decay,
                amsbound=self.amsgrad,
            )
        elif optimizer_name == "lamb":
            optimizer = Lamb(
                params,
                lr=self.learning_rate,
                betas=tuple(self.betas),
                eps=self.eps,
                weight_decay=self.weight_decay,
            )
        elif optimizer_name == "radam":
            optimizer = RAdam(
                params,
                lr=self.learning_rate,
                betas=tuple(self.betas),
                eps=self.eps,
                weight_decay=self.weight_decay,
            )
        else:
            Exception(
                "Invalid OPTIMIZER, try: 'adam', 'sgd', 'adabound', 'lamb' or 'radam'"
            )
        return optimizer
