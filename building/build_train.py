


def build_net(name:str,args):
    if name=='v1':
        print('v1 版本')
        from config.v1 import net_param
        from Match.v1.loss import build_criterion
        from Net.v1.yolo import  build_yolo


        model=build_yolo(net_param,args.device,args.conf,args.nms,True)
        critertion=build_criterion(net_param,args)
        return model,critertion
    
    elif name=='v2':
        print('v2 版本')
        from config.v2 import net_param
        from Match.v2.loss import build_criterion
        from Net.v2.yolo import build_yolo

        model=build_yolo(net_param,args.device,args.conf,args.nms,
                         args.topk,True)
        critertion=build_criterion(net_param,args)
    
    return model,critertion


    