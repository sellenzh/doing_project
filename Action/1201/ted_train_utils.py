import torch
import numpy as np
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

def binary_acc(pred, test):
    pred_tag = torch.round(pred)

    correct_results_sum = (pred_tag == test).sum().float()
    acc = correct_results_sum / test.shape[0]
    return acc


def train(model, train_loader, valid_loader, class_critirion, endp_critirion, 
            optimizer, checkpoint_filepath, writer, args):
    best_valid_loss = np.inf
    improvement_ratio = 0.01
    num_steps_wo_improvement = 0

    for epoch in range(args['epochs']):
        nb_batches_train = len(train_loader)
        train_acc = 0
        model.train()
        endp_losses = 0.0
        class_losses = 0.0
        full_losses = 0.0

        for data_enc, traj, y in train_loader:
            data_enc = data_enc.to(device)
            y = y.reshape(-1, 1).to(device)
            traj_in = traj[:, :-1].to(device)
            endp_real = traj[:, -1].to(device)

            endp, act, sigma_cls, sigma_endp = model(data_enc, traj_in)
            
            cl_loss = class_critirion(act, y)
            endp_loss = endp_critirion(endp, endp_real)
            f_loss = cl_loss / (sigma_cls * sigma_cls) + endp_loss / (sigma_endp * sigma_endp) + torch.log(sigma_cls) + torch.log(sigma_endp)

            model.zero_grad()
            f_loss.backward()

            full_losses += f_loss.item()
            class_losses += cl_loss.item()
            endp_losses += endp_loss.item()

            optimizer.step()

            train_acc += binary_acc(torch.round(act), y)

        draft_endpoint(endp, endp_real, epoch)

        print("sigma_cls: " + str(sigma_cls.item()))
        print("sigma_endp: " + str(sigma_endp.item()) + "\n")

        writer.add_scalar('training Full loss', full_losses / nb_batches_train, epoch + 1)
        writer.add_scalar('training Class loss', class_losses / nb_batches_train, epoch + 1) 
        writer.add_scalar('training EndPoint loss', endp_losses / nb_batches_train, epoch + 1)        
        writer.add_scalar('training Acc', train_acc / nb_batches_train, epoch + 1)

        print(f"Epoch {epoch}: | Train_Full_Loss {full_losses / nb_batches_train} | Train_Class_Loss {class_losses / nb_batches_train} | Train_Endp_Loss {endp_losses / nb_batches_train} | Train_Acc {train_acc / nb_batches_train} ")
        valid_full_loss, valid_class_loss, valid_endp_loss, val_acc = evaluate(model, valid_loader,
                                                                              class_critirion, endp_critirion, args)
        
        writer.add_scalar('validation Full loss', valid_full_loss, epoch + 1)
        writer.add_scalar('validation Class loss', valid_class_loss, epoch + 1) 
        writer.add_scalar('validation Endpoint loss', valid_endp_loss, epoch + 1) 
        writer.add_scalar('validation Acc', val_acc, epoch + 1)
        
        if (best_valid_loss - valid_class_loss) > np.abs(best_valid_loss * improvement_ratio):
            num_steps_wo_improvement = 0
        else:
            num_steps_wo_improvement += 1
            
        if num_steps_wo_improvement == 10:
            print("Early stopping on epoch:{}".format(str(epoch)))
            break
        if valid_class_loss <= best_valid_loss:
            best_valid_loss = valid_class_loss  
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'LOSS': full_losses / nb_batches_train,
            }, checkpoint_filepath)

def evaluate(model, data_loader, cls_critirion, endp_critirion):
    nb_batches = len(data_loader)
    val_full_losses = 0.0
    val_class_losses = 0.0
    val_endp_losses = 0.0
    with torch.no_grad():
        model.eval()
        acc = 0
        for data_enc, traj, y in data_loader:
            data_enc = data_enc.to(device)
            y = y.reshape(-1, 1).to(device)
            traj_in = traj[:, :-1].ro(device)
            endp_real = traj[:, -1].to(device)

            endp, act, sigma_cls, sigma_endp = model(data_enc)
            
            val_cl_loss = cls_critirion(act, y)
            val_end_loss = endp_critirion(endp, endp_real)
            val_f_loss = val_cl_loss / (sigma_cls * sigma_cls) + val_end_loss / (sigma_endp * sigma_endp) +\
                        torch.log(sigma_cls) + torch.log(sigma_endp)

            val_full_losses += val_f_loss.item()
            val_class_losses += val_cl_loss.item()
            val_endp_losses += val_end_loss.item()

            acc += binary_acc(torch.round(act), y)
    print(f"Valid_Full_Loss {val_full_losses / nb_batches} | Valid_Class_Loss {val_class_losses / nb_batches} | Valid_Endp_Loss {val_endp_losses / nb_batches} | Valid_Acc {acc / nb_batches} \n")
    return val_full_losses / nb_batches, val_class_losses / nb_batches, val_endp_losses / nb_batches, acc / nb_batches


def test(model, data_loader):
    with torch.no_grad():
        model.eval()
        step = 0
        for data_enc, traj, y in data_loader:
            data_enc = data_enc.to(device)
            y = y.reshape(-1, 1).to(device)
            #traj_in = traj[:, :-1].to(device)

            _, act, _, _ = model(data_enc)

            if(step == 0):
                pred = act
                labels = y

            else:
                pred = torch.cat((pred, act), 0)
                labels = torch.cat((labels, y), 0)
            step += 1

    return pred, labels


def draft_endpoint(pred, real, epoch):
    batch_idx = pred.shape[0]
    pred, real = pred.cpu().detach().numpy(), real.cpu().detach().numpy()
    plt.clf()

    for i in range(batch_idx):
        X = list((pred[i, 0], real[i, 0]))
        Y = list((pred[i, 1], real[i, 1]))
        plt.plot(X, Y)
    #plt.legend(loc=0)
    plt.xlabel('x_steps')
    plt.ylabel('y_steps')
    plt.title('pred and real endpoint\'s distance.')
    plt.grid(True, linestyle='--', alpha=1)
    plt.savefig(fname='./save_img/' + str(epoch) + '.jpg')
