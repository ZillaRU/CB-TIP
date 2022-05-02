import torch.nn as nn

def select_loss_function(loss_select):
    return {
        'ERROR': nn.BCEWithLogitsLoss(reduction='none'),  # if loss_select == 'focal' else BCEFocalLoss(),
        'DIFF': nn.KLDivLoss(reduction='none')
    }
