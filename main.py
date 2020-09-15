import os
import argparse
import numpy as np
from paddle import fluid
from visualdl import LogWriter
from net import MyNet
from const import MODEL_PATH, LOG_PATH, CHECKPOINT_PATH


log_writer = LogWriter(logdir=LOG_PATH)


def load_data(data_path, mode='train'):
    def generator():
        yield []

    return generator


def do_train(data_path, model_name='mymodel', use_gpu=False, epoch_num=5, batch_size=100, learning_rate=0.01):
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        model = MyNet()
        model.train()
        train_loader = load_data(data_path, mode='train')

        optimizer = fluid.optimizer.SGDOptimizer(
            learning_rate=learning_rate, parameter_list=model.parameters())

        iter = 0
        for epoch_id in range(epoch_num):
            for batch_id, data in enumerate(train_loader()):
                #准备数据，格式需要转换成符合框架要求的
                image_data, label_data = data
                # 将数据转为飞桨动态图格式
                image = fluid.dygraph.to_variable(image_data)
                label = fluid.dygraph.to_variable(label_data)
                
                # #前向计算的过程
                # predict = model(image)
                #前向计算的过程，同时拿到模型输出值和分类准确率
                predict, avg_acc = model(image, label)
                
                #计算损失，取一个批次样本损失的平均值
                # loss = fluid.layers.square_error_cost(predict, label)
                loss = fluid.layers.cross_entropy(predict, label)
                avg_loss = fluid.layers.mean(loss)
                
                #每训练了1000批次的数据，打印下当前Loss的情况
                if batch_id !=0 and batch_id  % 100 == 0:
                    print("epoch: {}, batch: {}, loss is: {}, acc is: {}".format(epoch_id, batch_id, avg_loss.numpy(), avg_acc.numpy()))
                    log_writer.add_scalar(tag = 'acc', step = iter, value = avg_acc.numpy())
                    log_writer.add_scalar(tag = 'loss', step = iter, value = avg_loss.numpy())
                    iter = iter + 100
                
                #后向传播，更新参数的过程
                avg_loss.backward()
                optimizer.minimize(avg_loss)
                model.clear_gradients()

            fluid.save_dygraph(model.state_dict(), os.path.join(CHECKPOINT_PATH, f'{model_name}_epoch_{epoch_id}'))
            fluid.save_dygraph(optimizer.state_dict(), os.path.join(CHECKPOINT_PATH, f'{model_name}_epoch_{epoch_id}'))

        # 保存模型
        fluid.save_dygraph(model.state_dict(), os.path.join(MODEL_PATH, model_name))


def do_eval(data_path, model_name='mymodel', use_gpu=False):
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        model = MyNet()
        model_state_dict, _ = fluid.load_dygraph(os.path.join(MODEL_PATH, model_name))
        model.load_dict(model_state_dict)

        model.eval()
        eval_loader = load_data(data_path, mode='eval')

        avg_acc_set = []
        avg_loss_set = []
        for _, data in enumerate(eval_loader()):
            x_data, y_data = data
            img = fluid.dygraph.to_variable(x_data)
            label = fluid.dygraph.to_variable(y_data)
            predict, avg_acc = model(img, label)
            loss = fluid.layers.cross_entropy(input=predict, label=label)
            avg_loss = fluid.layers.mean(loss)
            avg_acc_set.append(float(avg_acc.numpy()))
            avg_loss_set.append(float(avg_loss.numpy()))
        
        #计算多个batch的平均损失和准确率
        avg_acc_val_mean = np.array(avg_acc_set).mean()
        avg_loss_val_mean = np.array(avg_loss_set).mean()

        print('loss={}, acc={}'.format(avg_loss_val_mean, avg_acc_val_mean))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True, type=str, help='data path.')
    parser.add_argument('--model-name', default='mymodel', type=str, help='setup an model name.')
    parser.add_argument('--train-only', default=False, type=bool, help='run train only. default: False')
    parser.add_argument('--eval-only', default=False, type=bool, help='run eval only. default: False')
    parser.add_argument('--use-gpu', default=False, type=bool, help='use gpu. default: False')
    parser.add_argument('--epoch-num', default=5, type=int, help='setup epoch num. default: 5')
    parser.add_argument('--batch-size', default=100, type=int, help='setup batch size. default: 100')
    parser.add_argument('--learning-rate', default=0.01, type=float, help='setup batch size. default: 0.01')
    args = parser.parse_args()

    if args.train_only is False and args.eval_only is False:
        do_train(args.data_path, args.model_name, args.use_gpu, args.epoch_num, args.batch_size, args.learning_rate)
        do_eval(args.data_path, args.model_name, args.use_gpu)
    
    if args.train_only is True:
        do_train(args.data_path, args.model_name, args.use_gpu, args.epoch_num, args.batch_size, args.learning_rate)

    if args.eval_only is True:
        do_eval(args.data_path, args.model_name, args.use_gpu)


if __name__ == "__main__":
    main()