import torch


def model_metric(x, y, model):

    """
    Calculate distance between x and y in model space root mean squared error

    Parameters
    -----------
    image: torch.Tensor
        image, (B x C x H x W)
    model: torch class
        torch model with defined forward and backward operations

    Notes
    -----


    """

    repx = model(x)
    repy = model(y)

    # for optimization purpose (stabilizing the gradient around zero)
    epsilon = 1e-10

    dist = torch.sqrt(torch.mean((repx - repy) ** 2) + epsilon)

    return dist


# in case an object oriented implementation makes more sense

# class ModelMetric(nn.Module):
#
#     def __init__(self, model):
#         super(ModelMetric, self).__init__()
#         """
#         Calculate distance between x and y in model space root mean squared error
#
#         Parameters
#         -----------
#         image: torch.Tensor
#             image, (B x C x H x W)
#         model: torch class
#             torch model with defined forward and backward operations
#
#         Notes
#         -----
#
#
#         """
#
#         self.model = simulate.model
#
#     def forward(self, x, y):
#
#         repx = self.model(x)
#         repy = self.model(y)
#
#         # for optimization purpose (stabilizing the gradient around zero)
#         epsilon = 1e-10
#
#         dist = torch.sqrt(torch.mean((repx - repy) ** 2) + epsilon)
#
#         return dist