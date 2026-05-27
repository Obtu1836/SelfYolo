
from config.v3 import logger


def build_net(args, device: str, is_train: bool = True):
    if args.version == 'v1':
        logger.info('v1 版本')
        from config.v1 import net_param
        from Match.v1.loss import build_criterion
        from Net.v1.yolo import build_yolo

        model = build_yolo(net_param,args.img_size, device, args.conf, args.nms, is_train)
        critertion = build_criterion(net_param, args)
        return model, critertion

    elif args.version == 'v2':
        logger.info('v2 版本')
        from config.v2 import net_param
        from Match.v2.loss import build_criterion
        from Net.v2.yolo import build_yolo

        model = build_yolo(net_param, device, args.conf, args.nms,
                           args.topk, is_train)
        critertion = build_criterion(net_param, args)

    elif args.version == 'v3':
        logger.info('v3版本')
        from config.v3 import net_param
        from Match.v3.loss import build_criterion
        from Net.v3.yolo import build_yolo

        model = build_yolo(net_param, args.img_size, device,
                           args.conf, args.nms, args.topk, is_train)

        critertion = build_criterion(weights=net_param.weights,
                                     num_class=net_param.num_class,
                                     num_anchors=net_param.num_anchors,
                                     anchor_size=net_param.anchor_size,
                                     anchor_base_size=net_param.anchor_base_size)

    return model, critertion
