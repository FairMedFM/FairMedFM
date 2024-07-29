import torch
from models.utils import get_model
from wrappers.utils import get_warpped_model

def get_trainer(args, model, logger, test_dataloader=None):
    if args.task == "cls":
        if args.method == "erm":
            from trainers import CLSTrainer
            return CLSTrainer(args, model, test_dataloader, logger)
        elif args.method == "sa":
            from trainers import SATrainer
            return SATrainer(args, model, test_dataloader, logger)
        else:
            raise NotImplementedError
    elif args.task == "seg":
        if args.method == "erm":
            from trainers import SegTrainer
            return SegTrainer(args, model, logger)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
