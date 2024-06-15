def get_trainer(args, model, logger):
    if args.task == "cls":
        if args.method == "erm":
            from trainers import CLSTrainer

            return CLSTrainer(args, model, logger)
    elif args.tasl == "seg":
        # TODO
        pass
    else:
        raise NotImplementedError
