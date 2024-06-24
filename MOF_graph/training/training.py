##General imports
import csv
import os
import time
from datetime import datetime
import shutil
import copy
import numpy as np
from functools import partial
import platform

from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan

##Torch imports
import torch.nn.functional as F
import torch
#from torch_geometric.data import DataLoader, Dataset
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DataParallel
import torch_geometric.transforms as T
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as mp

##MOF_graph imports
from MOF_graph import models
import MOF_graph.process as process
from MOF_graph import training as training
from MOF_graph.models.utils import model_summary
#from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
################################################################################
#  Training functions
################################################################################

##Train step, runs model in train mode
def train(model, optimizer, loader, loss_method, rank):
    model.train()
    loss_all = 0
    count = 0
    for data in loader:
        # print(rank)
        data = data.to('cuda:1')
        optimizer.zero_grad()
        output = model(data)
        #data.y = torch.reshape(data.y,(-1,output.size(-1)))
        #data.y = torch.reshape(data.y,(-1,output.size(1)))
        if count == 0:
            output_all = output
            y_all = data.y
        else:
            output_all = torch.cat((output_all,output),0)
            y_all = torch.cat((y_all, data.y),0)
        #print(data.y.shape, output.shape)
        #print(data.y)
        #print(output)
        loss = getattr(F, loss_method)(output, data.y)
        loss.backward()
        loss_all += loss.detach() * output.size(0)
        
        
            
        # clip = 10
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

        optimizer.step()
        count = count + output.size(0)
    # print(y_all.shape, output_all.shape)
    #R2 = r2_score(y_all.data.cpu().numpy(),output_all.data.cpu().numpy())
    R2 = r2_score(y_all, output_all)
    #MSE = mean_squared_error(y_all, output_all)
    #RMSE = MSE**0.5
    RMSE = rmse_score(y_all, output_all)
    loss_all = loss_all / count
    return loss_all, R2, RMSE

def r2_score(y_true, y_pred):
    y_mean = torch.mean(y_true, dim=0)
    total_sum_of_squares = torch.sum((y_true - y_mean)**2, dim=0)
    residual_sum_of_squares = torch.sum((y_true - y_pred)**2, dim=0)
    r2 = 1 - (residual_sum_of_squares / total_sum_of_squares)
    # print(r2)
    return r2

def rmse_score(y_true, y_pred):
    diff = y_pred - y_true
    mse = torch.mean(diff ** 2)
    rmse = torch.sqrt(mse)
    return rmse

##Evaluation step, runs model in eval mode
def evaluate(loader, model, loss_method, rank, out=False):
    model.eval()
    loss_all = 0
    count = 0
    for data in loader:
        data = data.to('cuda:1')
        with torch.no_grad():
            output = model(data)
            # data.y = torch.reshape(data.y,(-1, output.size(1)))
            # print(output.size())
            loss = getattr(F, loss_method)(output, data.y)
            loss_all += loss * output.size(0)
            #if out == True:
            if count == 0:
                ids = [sublist for sublist in data.structure_id]
                predict_result = output#.data.cpu().numpy()
                target = data.y#.cpu().numpy()
                    
            else:
                ids_temp = [
                        sublist for sublist in data.structure_id
                    ]
                #ids_temp = [item for sublist in ids_temp for item in sublist]
                ids = ids + ids_temp
                predict_result = torch.cat(
                        (predict_result, output), axis=0
                    )
                target = torch.cat((target, data.y), axis=0)
            count = count + output.size(0)
            
    loss_all = loss_all / count
    
    if out == True:
        test_out = np.column_stack((ids, target.data.cpu().numpy(), predict_result.data.cpu().numpy()))
        R2 = r2_score(target,predict_result)
        RMSE = rmse_score(target,predict_result)
        return loss_all, test_out, R2, RMSE
    elif out == False:
        R2 = r2_score(target,predict_result)
        RMSE = rmse_score(target,predict_result)
        return loss_all, R2, RMSE


##Model trainer
def trainer(
    rank,
    world_size,
    model,
    optimizer,
    scheduler,
    loss,#l1_loss
    train_loader,
    val_loader,
    train_sampler,
    epochs,
    verbosity,#5
    filename = "my_model_temp.pth",
):

    train_error = val_error = test_error = epoch_time = float("NaN")
    train_start = time.time()
    best_val_error = 1e10
    model_best = model
    ##Start training over epochs loop
    for epoch in range(1, epochs + 1):
        #print(epoch)
        lr = scheduler.optimizer.param_groups[0]["lr"]
        if rank not in ("cpu", "cuda"):
            train_sampler.set_epoch(epoch)
        ##Train model
        train_error, R2_train, RMSE_train = train(model, optimizer, train_loader, loss, rank=rank)
       # print("finish train")
        if rank not in ("cpu", "cuda"):
            torch.distributed.reduce(train_error, dst=0)
            train_error = train_error / world_size

        ##Get validation performance
        if rank not in ("cpu", "cuda"):
            dist.barrier()
        if val_loader != None and rank in (0, "cpu", "cuda"):
            if rank not in ("cpu", "cuda"):
                val_error, R2_val, RMSE_val = evaluate(
                    val_loader, model.module, loss, rank=rank, out=False
                )
            else:
                val_error, R2_val, RMSE_val = evaluate(val_loader, model, loss, rank=rank, out=False)

        ##Train loop timings
        epoch_time = time.time() - train_start
        train_start = time.time()
        allocated_memory = torch.cuda.memory_allocated(device = 'cuda:1')
        cached_memory = torch.cuda.memory_cached(device = 'cuda:1')
        
        #print("mode2")
        ##remember the best val error and save model and checkpoint        
        if val_loader != None and rank in (0, "cpu", "cuda"):
            if val_error == float("NaN") or val_error < best_val_error:
                if rank not in ("cpu", "cuda"):
                    model_best = copy.deepcopy(model.module)
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "full_model": model,
                        },
                        filename,
                    )
                else:
                    print(filename)
                    model_best = copy.deepcopy(model)
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "full_model": model,
                        },
                        filename,
                    )
            best_val_error = min(val_error, best_val_error)
           # print("mode3")
        elif val_loader == None and rank in (0, "cpu", "cuda"):
            if rank not in ("cpu", "cuda"):
                model_best = copy.deepcopy(model.module)
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "full_model": model,
                    },
                    filename,
                )
            else:
                # checkpoint_dir = os.path.join("./save_model")
                model_best = copy.deepcopy(model)
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "full_model": model,
                    },
                    filename,
                )
                # to_save = {
                #     "model": model,
                #     "optimizer": optimizer,
                #     "lr_scheduler": scheduler,
                #     "trainer": trainer,
                # }
                # handler = Checkpoint(
                #     to_save,
                #     DiskSaver(checkpoint_dir, create_dir=True, require_empty=False),
                #     n_saved=2,
                #     global_step_transform=lambda *_: trainer.state.epoch,
                # )
                # trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)
                # # evaluate save
                # to_save = {"model": net}
                # handler = Checkpoint(
                #     to_save,
                #     DiskSaver(checkpoint_dir, create_dir=True, require_empty=False),
                #     n_saved=5,
                #     filename_prefix='best',
                #     score_name="neg_mae",
                #     global_step_transform=lambda *_: trainer.state.epoch,
                # )
                # evaluator.add_event_handler(Events.EPOCH_COMPLETED, handler)

        ##scheduler on train error
        scheduler.step(train_error)
        #print('mode4')
        ##Print performance
        if epoch % verbosity == 0:
            if rank in (0, "cpu", "cuda"):
                #R2_train_print = ', '.join(map(str, R2_train.tolist()))
                #R2_val_print = ', '.join(map(str, R2_val.tolist()))
                print(
                    "Epoch: {:04d}, Learning Rate: {:.6f}, Training Error: {:.5f}, Val Error: {:.5f}, R2_train: {}, R2_val: {}, RMSE_train: {}, RMSE_val: {}, Time per epoch (s): {:.5f}, allocated_memory:{}, cached_memory:{}".format(
                        epoch, lr, train_error, val_error, R2_train, R2_val, RMSE_train, RMSE_val, epoch_time,allocated_memory, cached_memory
                    )
                )

    if rank not in ("cpu", "cuda"):
        dist.barrier()

    return model_best


##Write results to csv file
def write_results(output, filename):
    shape = output.shape
    with open(filename, "w",newline="") as f:
        csvwriter = csv.writer(f)
        for i in range(0, len(output)):
            if i == 0:
                csvwriter.writerow(
                    ["ids"]
                    + ["target"] * int((shape[1] - 1) / 2)
                    + ["prediction"] * int((shape[1] - 1) / 2)
                )
            elif i > 0:
                csvwriter.writerow(output[i - 1, :])


##Pytorch ddp setup
def ddp_setup(rank, world_size):
    if rank in ("cpu", "cuda"):
        return
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    if platform.system() == 'Windows':
        dist.init_process_group("gloo", rank=rank, world_size=world_size)    
    else:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = True


##Pytorch model setup
def model_setup(
    rank,
    model_name,
    model_params,
    dataset,
    load_model=False,
    model_path=None,
    print_model=True,
):
    model = getattr(models, model_name)(
        data=dataset, **(model_params if model_params is not None else {})
    ).to('cuda:1')
    if load_model == "True":
        assert os.path.exists(model_path), "Saved model not found"
        if str(rank) in ("cpu"):
            saved = torch.load(model_path, map_location=torch.device("cpu"))
        else:
            saved = torch.load(model_path)
        model.load_state_dict(saved["model_state_dict"])
        # optimizer.load_state_dict(saved['optimizer_state_dict'])

    # DDP
    if rank not in ("cpu", "cuda"):
        model = DistributedDataParallel(
            model, device_ids=[rank], find_unused_parameters=True
        )
        # model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=False)
    if print_model == True and rank in (0, "cpu", "cuda"):
        model_summary(model)
    return model


##Pytorch loader setup
def loader_setup(
    train_ratio,
    val_ratio,
    test_ratio,
    batch_size,
    dataset,
    rank,
    seed,
    world_size=0,
    num_workers=0,
):
    ##Split datasets
    train_dataset, val_dataset, test_dataset = process.split_data(
        dataset, train_ratio, val_ratio, test_ratio, seed
    )

    ##DDP
    if rank not in ("cpu", "cuda"):
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank
        )
    elif rank in ("cpu", "cuda"):
        train_sampler = None

    ##Load data
    train_loader = val_loader = test_loader = None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,#(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    # may scale down batch size if memory is an issue
    if rank in (0, "cpu", "cuda"):
        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=True,#False,
                num_workers=num_workers,
                pin_memory=True,
            )
        if len(test_dataset) > 0:
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=True,#False,
                num_workers=num_workers,
                pin_memory=True,
            )
    return (
        train_loader,
        val_loader,
        test_loader,
        train_sampler,
        train_dataset,
        val_dataset,
        test_dataset,
    )

#def normalize_label(label):

def loader_setup_CV(index, batch_size, dataset, rank, world_size=0, num_workers=0):
    ##Split datasets
    train_dataset = [x for i, x in enumerate(dataset) if i != index]
    train_dataset = torch.utils.data.ConcatDataset(train_dataset)
    test_dataset = dataset[index]

    ##DDP
    if rank not in ("cpu", "cuda"):
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank
        )
    elif rank in ("cpu", "cuda"):
        train_sampler = None

    train_loader = val_loader = test_loader = None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,#(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    if rank in (0, "cpu", "cuda"):
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True,#False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, test_loader, train_sampler, train_dataset, test_dataset


################################################################################
#  Trainers
################################################################################

###Regular training with train, val, test split
def train_regular(
    rank,
    world_size,
    data_path,#data/MOF_data/MOF_data
    job_parameters=None,#config["Job"],
    training_parameters=None,#config["Training"],
    model_parameters=None,#config["Models"]
):
    ##DDP
    ddp_setup(rank, world_size)
    ##some issues with DDP learning rate
    if rank not in ("cpu", "cuda"):
        model_parameters["lr"] = model_parameters["lr"] * world_size

    ##Get dataset target_index=0
    dataset = process.get_dataset(data_path, training_parameters["target_index"], False)

    if rank not in ("cpu", "cuda"):
        dist.barrier()

    ##Set up loader
    ##
    #Index of target column in targets.csv
    #    target_index: 0
    #Loss functions (from pytorch) examples: l1_loss, mse_loss, binary_cross_entropy
    #    loss: "l1_loss"       
    #Ratios for train/val/test split out of a total of 1  
    #    train_ratio: 0.8
    #    val_ratio: 0.05
    #    test_ratio: 0.15
    #Training print out frequency (print per n number of epochs)
    #    verbosity: 5
    (
        train_loader,
        val_loader,
        test_loader,
        train_sampler,
        train_dataset,
        _,
        _,
    ) = loader_setup(
        training_parameters["train_ratio"],#0.8
        training_parameters["val_ratio"],  #0.05
        training_parameters["test_ratio"], #0.15
        model_parameters["batch_size"],    #100
        dataset,
        rank,
        job_parameters["seed"],
        world_size,
    )

    ##Set up model
    model = model_setup(
        rank,#world_size
        model_parameters["model"],
        model_parameters,
        dataset,
        job_parameters["load_model"],#False
        job_parameters["model_path"],#my_model.pth
        model_parameters.get("print_model", True),
    )

    ##Set-up optimizer & scheduler
    optimizer = getattr(torch.optim, model_parameters["optimizer"])(
        model.parameters(),
        lr=model_parameters["lr"],
        **model_parameters["optimizer_args"]
    )
    scheduler = getattr(torch.optim.lr_scheduler, model_parameters["scheduler"])(
        optimizer, **model_parameters["scheduler_args"]
    )

    ##Start training
    model = trainer(
        rank,
        world_size,
        model,
        optimizer,
        scheduler,
        training_parameters["loss"],
        train_loader,
        val_loader,
        train_sampler,
        model_parameters["epochs"],
        training_parameters["verbosity"],
        model_parameters["filename"], #"my_model_temp.pth",
    )

    if rank in (0, "cpu", "cuda"):

        train_error = val_error = test_error = float("NaN")

        ##workaround to get training output in DDP mode
        ##outputs are slightly different, could be due to dropout or batchnorm?
        train_loader = DataLoader(
            train_dataset,
            batch_size=model_parameters["batch_size"],
            shuffle=True,#False,
            num_workers=0,
            pin_memory=True,
        )
        #

        ##Get train error in eval mode
        train_error, train_out, R2_Train, RMSE_Train = evaluate(
            train_loader, model, training_parameters["loss"], rank, out=True
        )
        #R2_Train_print = ', '.join(map(str, R2_Train.tolist()))
        print("Train Error: {:.5f}".format(train_error), "Train R^2: {}".format(R2_Train), "Train RMSE: {}".format(RMSE_Train))

        ##Get val error
        if val_loader != None:
            val_error, val_out, R2_Val, RMSE_Val = evaluate(
                val_loader, model, training_parameters["loss"], rank, out=True
            )
            #R2_Val_print = ', '.join(map(str, R2_Val.tolist()))
            print("Val Error: {:.5f}".format(val_error), "Val R^2: {}".format(R2_Val), "Val RMSE: {}".format(RMSE_Val))

        ##Get test error
        if test_loader != None:
            test_error, test_out, R2_Test, RMSE_Test = evaluate(
                test_loader, model, training_parameters["loss"], rank, out=True
            )
            #R2_Test_print = ', '.join(map(str, R2_Test.tolist()))
            print("Test Error: {:.5f}".format(test_error), "Test R^2: {}".format(R2_Test), "Test RMSE: {}".format(RMSE_Test))

        ##Save model
        if job_parameters["save_model"] == "True":

            if rank not in ("cpu", "cuda"):
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "full_model": model,
                    },
                    job_parameters["model_path"],
                )
            else:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "full_model": model,
                    },
                    job_parameters["model_path"],
                )

        ##Write outputs
        if job_parameters["write_output"] == "True":

            write_results(
                train_out, str(model_parameters["filename"]).split('_')[0] + '_' + str(job_parameters["job_name"]) + "_train_outputs.csv"
            )
            if val_loader != None:
                write_results(
                    val_out, str(model_parameters["filename"]).split('_')[0] + '_' + str(job_parameters["job_name"]) + "_val_outputs.csv"
                )
            if test_loader != None:
                write_results(
                    test_out, str(model_parameters["filename"]).split('_')[0] + '_' + str(job_parameters["job_name"]) + "_test_outputs.csv"
                )

        if rank not in ("cpu", "cuda"):
            dist.destroy_process_group()

        ##Write out model performance to file
        error_values = np.array((train_error.cpu(), val_error.cpu(), test_error.cpu()))
        if job_parameters.get("write_error") == "True":
            np.savetxt(
                job_parameters["job_name"] + "_errorvalues.csv",
                error_values[np.newaxis, ...],
                delimiter=",",
            )

        return error_values


###Predict using a saved movel
def predict(dataset, loss, job_parameters=None):

    rank = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    ##Loads predict dataset in one go, care needed for large datasets)
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,#False,
        num_workers=0,
        pin_memory=True,
    )

    ##Load saved model
    assert os.path.exists(job_parameters["model_path"]), "Saved model not found"
    if str(rank) == "cpu":
        saved = torch.load(
            job_parameters["model_path"], map_location=torch.device("cpu")
        )
    else:
        saved = torch.load(
            job_parameters["model_path"], map_location=torch.device("cuda:1")
        )
    model = saved["full_model"]
    model = model.to(rank)
    model_summary(model)

    ##Get predictions
    time_start = time.time()
    test_error, test_out, R2_Test, RMSE_Test = evaluate(loader, model, loss, rank, out=True)
    elapsed_time = time.time() - time_start

    print("Evaluation time (s): {:.5f}".format(elapsed_time))

    ##Write output
    if job_parameters["write_output"] == "True":
        write_results(
            test_out, str(job_parameters["job_name"]) + "_predicted_outputs.csv"
        )

    return test_error,R2_Test, RMSE_Test


###n-fold cross validation
def train_CV(
    rank,
    world_size,
    data_path,
    job_parameters=None,
    training_parameters=None,
    model_parameters=None,
):

    job_parameters["load_model"] = "False"
    job_parameters["save_model"] = "False"
    job_parameters["model_path"] = None
    ##DDP
    ddp_setup(rank, world_size)
    ##some issues with DDP learning rate
    if rank not in ("cpu", "cuda"):
        model_parameters["lr"] = model_parameters["lr"] * world_size

    ##Get dataset
    dataset = process.get_dataset(data_path, training_parameters["target_index"], False)

    ##Split datasets
    cv_dataset = process.split_data_CV(
        dataset, num_folds=job_parameters["cv_folds"], seed=job_parameters["seed"]
    )
    cv_error = 0

    for index in range(0, len(cv_dataset)):

        ##Set up model
        if index == 0:
            model = model_setup(
                rank,
                model_parameters["model"],
                model_parameters,
                dataset,
                job_parameters["load_model"],
                job_parameters["model_path"],
                print_model=True,
            )
        else:
            model = model_setup(
                rank,
                model_parameters["model"],
                model_parameters,
                dataset,
                job_parameters["load_model"],
                job_parameters["model_path"],
                print_model=False,
            )

        ##Set-up optimizer & scheduler
        optimizer = getattr(torch.optim, model_parameters["optimizer"])(
            model.parameters(),
            lr=model_parameters["lr"],
            **model_parameters["optimizer_args"]
        )
        scheduler = getattr(torch.optim.lr_scheduler, model_parameters["scheduler"])(
            optimizer, **model_parameters["scheduler_args"]
        )

        ##Set up loader
        train_loader, test_loader, train_sampler, train_dataset, _ = loader_setup_CV(
            index, model_parameters["batch_size"], cv_dataset, rank, world_size
        )

        ##Start training
        model = trainer(
            rank,
            world_size,
            model,
            optimizer,
            scheduler,
            training_parameters["loss"],
            train_loader,
            None,
            train_sampler,
            model_parameters["epochs"],
            training_parameters["verbosity"],
            "my_model_temp.pth",
        )

        if rank not in ("cpu", "cuda"):
            dist.barrier()

        if rank in (0, "cpu", "cuda"):

            train_loader = DataLoader(
                train_dataset,
                batch_size=model_parameters["batch_size"],
                shuffle=True,#False,
                num_workers=0,
                pin_memory=True,
            )

            ##Get train error
            train_error, train_out, R2_Train, RMSE_Train = evaluate(
                train_loader, model, training_parameters["loss"], rank, out=True
            )
            print("Train Error: {:.5f}".format(train_error))

            ##Get test error
            test_error, test_out, R2_Test, RMSE_Test = evaluate(
                test_loader, model, training_parameters["loss"], rank, out=True
            )
            print("Test Error: {:.5f}".format(test_error))

            cv_error = cv_error + test_error

            if index == 0:
                total_rows = test_out
            else:
                total_rows = np.vstack((total_rows, test_out))

    ##Write output
    if rank in (0, "cpu", "cuda"):
        if job_parameters["write_output"] == "True":
            if test_loader != None:
                write_results(
                    total_rows, str(job_parameters["job_name"]) + "_CV_outputs.csv"
                )

        cv_error = cv_error / len(cv_dataset)
        print("CV Error: {:.5f}".format(cv_error))

    if rank not in ("cpu", "cuda"):
        dist.destroy_process_group()

    return cv_error


### Repeat training for n times
def train_repeat(
    data_path,
    job_parameters=None,
    training_parameters=None,
    model_parameters=None,
):

    world_size = torch.cuda.device_count()
    job_name = job_parameters["job_name"]
    model_path = job_parameters["model_path"]
    job_parameters["write_error"] = "True"
    job_parameters["load_model"] = "False"
    job_parameters["save_model"] = "False"
    ##Loop over number of repeated trials
    for i in range(0, job_parameters["repeat_trials"]):

        ##new seed each time for different data split
        job_parameters["seed"] = np.random.randint(1, 1e6)

        if i == 0:
            model_parameters["print_model"] = True
        else:
            model_parameters["print_model"] = False

        job_parameters["job_name"] = job_name + str(i)
        job_parameters["model_path"] = str(i) + "_" + model_path

        if world_size == 0:
            print("Running on CPU - this will be slow")
            training.train_regular(
                "cpu",
                world_size,
                data_path,
                job_parameters,
                training_parameters,
                model_parameters,
            )
        elif world_size > 0:
            if job_parameters["parallel"] == "True":
                print("Running on", world_size, "GPUs")
                mp.spawn(
                    training.train_regular,
                    args=(
                        world_size,
                        data_path,
                        job_parameters,
                        training_parameters,
                        model_parameters,
                    ),
                    nprocs=world_size,
                    join=True,
                )
            if job_parameters["parallel"] == "False":
                print("Running on one GPU")
                training.train_regular(
                    "cuda",
                    world_size,
                    data_path,
                    job_parameters,
                    training_parameters,
                    model_parameters,
                )

    ##Compile error metrics from individual trials
    print("Individual training finished.")
    print("Compiling metrics from individual trials...")
    error_values = np.zeros((job_parameters["repeat_trials"], 3))
    for i in range(0, job_parameters["repeat_trials"]):
        filename = job_name + str(i) + "_errorvalues.csv"
        error_values[i] = np.genfromtxt(filename, delimiter=",")
    mean_values = [
        np.mean(error_values[:, 0]),
        np.mean(error_values[:, 1]),
        np.mean(error_values[:, 2]),
    ]
    std_values = [
        np.std(error_values[:, 0]),
        np.std(error_values[:, 1]),
        np.std(error_values[:, 2]),
    ]

    ##Print error
    print(
        "Training Error Avg: {:.3f}, Training Standard Dev: {:.3f}".format(
            mean_values[0], std_values[0]
        )
    )
    print(
        "Val Error Avg: {:.3f}, Val Standard Dev: {:.3f}".format(
            mean_values[1], std_values[1]
        )
    )
    print(
        "Test Error Avg: {:.3f}, Test Standard Dev: {:.3f}".format(
            mean_values[2], std_values[2]
        )
    )

    ##Write error metrics
    if job_parameters["write_output"] == "True":
        with open(job_name + "_all_errorvalues.csv", "w") as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(
                [
                    "",
                    "Training",
                    "Validation",
                    "Test",
                ]
            )
            for i in range(0, len(error_values)):
                csvwriter.writerow(
                    [
                        "Trial " + str(i),
                        error_values[i, 0],
                        error_values[i, 1],
                        error_values[i, 2],
                    ]
                )
            csvwriter.writerow(["Mean", mean_values[0], mean_values[1], mean_values[2]])
            csvwriter.writerow(["Std", std_values[0], std_values[1], std_values[2]])
    elif job_parameters["write_output"] == "False":
        for i in range(0, job_parameters["repeat_trials"]):
            filename = job_name + str(i) + "_errorvalues.csv"
            os.remove(filename)


###Hyperparameter optimization
# trainable function for ray tune (no parallel, max 1 GPU per job)
def tune_trainable(config, checkpoint_dir=None, data_path=None):

    # imports
    from ray import tune

    print("Hyperparameter trial start")
    hyper_args = config["hyper_args"]
    job_parameters = config["job_parameters"]
    processing_parameters = config["processing_parameters"]
    training_parameters = config["training_parameters"]
    model_parameters = config["model_parameters"]

    ##Merge hyperparameter parameters with constant parameters, with precedence over hyperparameter ones
    ##Omit training and job parameters as they should not be part of hyperparameter opt, in theory
    model_parameters = {**model_parameters, **hyper_args}
    processing_parameters = {**processing_parameters, **hyper_args}

    ##Assume 1 gpu or 1 cpu per trial, no functionality for parallel yet
    world_size = 1
    rank = "cpu"
    if torch.cuda.is_available():
        rank = "cuda"

    ##Reprocess data in a separate directory to prevent conflict
    if job_parameters["reprocess"] == "True":
        time = datetime.now()
        processing_parameters["processed_path"] = time.strftime("%H%M%S%f")
        processing_parameters["verbose"] = "False"
    data_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    )
    data_path = os.path.join(data_path, processing_parameters["data_path"])
    data_path = os.path.normpath(data_path)
    print("Data path", data_path)

    ##Set up dataset
    dataset = process.get_dataset(
        data_path,
        training_parameters["target_index"],
        job_parameters["reprocess"],
        processing_parameters,
    )

    ##Set up loader
    (
        train_loader,
        val_loader,
        test_loader,
        train_sampler,
        train_dataset,
        _,
        _,
    ) = loader_setup(
        training_parameters["train_ratio"],
        training_parameters["val_ratio"],
        training_parameters["test_ratio"],
        model_parameters["batch_size"],
        dataset,
        rank,
        job_parameters["seed"],
        world_size,
    )

    ##Set up model
    model = model_setup(
        rank,
        model_parameters["model"],
        model_parameters,
        dataset,
        False,
        None,
        False,
    )

    ##Set-up optimizer & scheduler
    optimizer = getattr(torch.optim, model_parameters["optimizer"])(
        model.parameters(),
        lr=model_parameters["lr"],
        **model_parameters["optimizer_args"]
    )
    scheduler = getattr(torch.optim.lr_scheduler, model_parameters["scheduler"])(
        optimizer, **model_parameters["scheduler_args"]
    )

    ##Load checkpoint
    if checkpoint_dir:
        model_state, optimizer_state, scheduler_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint")
        )
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(scheduler_state)

    ##Training loop
    for epoch in range(1, model_parameters["epochs"] + 1):
        lr = scheduler.optimizer.param_groups[0]["lr"]
        train_error = train(
            model, optimizer, train_loader, training_parameters["loss"], rank=rank
        )

        val_error = evaluate(
            val_loader, model, training_parameters["loss"], rank=rank, out=False
        )

        ##Delete processed data
        if epoch == model_parameters["epochs"]:
            if (
                job_parameters["reprocess"] == "True"
                and job_parameters["hyper_delete_processed"] == "True"
            ):
                shutil.rmtree(
                    os.path.join(data_path, processing_parameters["processed_path"])
                )
            print("Finished Training")

        ##Update to tune
        if epoch % job_parameters["hyper_iter"] == 0:
            with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save(
                    (
                        model.state_dict(),
                        optimizer.state_dict(),
                        scheduler.state_dict(),
                    ),
                    path,
                )
            ##Somehow tune does not recognize value without *1
            tune.report(loss=val_error.cpu().numpy() * 1)
            # tune.report(loss=val_error)


# Tune setup
def tune_setup(
    hyper_args,
    job_parameters,
    processing_parameters,
    training_parameters,
    model_parameters,
):

    # imports
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.suggest.hyperopt import HyperOptSearch
    from ray.tune.suggest import ConcurrencyLimiter
    from ray.tune import CLIReporter

    ray.init()
    data_path = "_"
    local_dir = "ray_results"
    # currently no support for paralleization per trial
    gpus_per_trial = 1

    ##Set up search algo
    search_algo = HyperOptSearch(metric="loss", mode="min", n_initial_points=5)
    search_algo = ConcurrencyLimiter(
        search_algo, max_concurrent=job_parameters["hyper_concurrency"]
    )

    ##Resume run
    if os.path.exists(local_dir + "/" + job_parameters["job_name"]) and os.path.isdir(
        local_dir + "/" + job_parameters["job_name"]
    ):
        if job_parameters["hyper_resume"] == "False":
            resume = False
        elif job_parameters["hyper_resume"] == "True":
            resume = True
        # else:
        #    resume = "PROMPT"
    else:
        resume = False

    ##Print out hyperparameters
    parameter_columns = [
        element for element in hyper_args.keys() if element not in "global"
    ]
    parameter_columns = ["hyper_args"]
    reporter = CLIReporter(
        max_progress_rows=20, max_error_rows=5, parameter_columns=parameter_columns
    )

    ##Run tune
    tune_result = tune.run(
        partial(tune_trainable, data_path=data_path),
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config={
            "hyper_args": hyper_args,
            "job_parameters": job_parameters,
            "processing_parameters": processing_parameters,
            "training_parameters": training_parameters,
            "model_parameters": model_parameters,
        },
        num_samples=job_parameters["hyper_trials"],
        # scheduler=scheduler,
        search_alg=search_algo,
        local_dir=local_dir,
        progress_reporter=reporter,
        verbose=job_parameters["hyper_verbosity"],
        resume=resume,
        log_to_file=True,
        name=job_parameters["job_name"],
        max_failures=4,
        raise_on_failed_trial=False,
        # keep_checkpoints_num=job_parameters["hyper_keep_checkpoints_num"],
        # checkpoint_score_attr="min-loss",
        stop={
            "training_iteration": model_parameters["epochs"]
            // job_parameters["hyper_iter"]
        },
    )

    ##Get best trial
    best_trial = tune_result.get_best_trial("loss", "min", "all")
    # best_trial = tune_result.get_best_trial("loss", "min", "last")

    return best_trial


###Simple ensemble using averages
def train_ensemble(
    data_path,
    job_parameters=None,
    training_parameters=None,
    model_parameters=None,
):

    world_size = torch.cuda.device_count()
    job_name = job_parameters["job_name"]
    write_output = job_parameters["write_output"]
    model_path = job_parameters["model_path"]
    job_parameters["write_error"] = "True"
    job_parameters["write_output"] = "True"
    job_parameters["load_model"] = "False"
    ##Loop over number of repeated trials
    for i in range(0, len(job_parameters["ensemble_list"])):
        job_parameters["job_name"] = job_name + str(i)
        job_parameters["model_path"] = (
            str(i) + "_" + job_parameters["ensemble_list"][i] + "_" + model_path
        )

        if world_size == 0:
            print("Running on CPU - this will be slow")
            training.train_regular(
                "cpu",
                world_size,
                data_path,
                job_parameters,
                training_parameters,
                model_parameters[job_parameters["ensemble_list"][i]],
            )
        elif world_size > 0:
            if job_parameters["parallel"] == "True":
                print("Running on", world_size, "GPUs")
                mp.spawn(
                    training.train_regular,
                    args=(
                        world_size,
                        data_path,
                        job_parameters,
                        training_parameters,
                        model_parameters[job_parameters["ensemble_list"][i]],
                    ),
                    nprocs=world_size,
                    join=True,
                )
            if job_parameters["parallel"] == "False":
                print("Running on one GPU")
                training.train_regular(
                    "cuda",
                    world_size,
                    data_path,
                    job_parameters,
                    training_parameters,
                    model_parameters[job_parameters["ensemble_list"][i]],
                )

    ##Compile error metrics from individual models
    print("Individual training finished.")
    print("Compiling metrics from individual models...")
    error_values = np.zeros((len(job_parameters["ensemble_list"]), 3))
    for i in range(0, len(job_parameters["ensemble_list"])):
        filename = job_name + str(i) + "_errorvalues.csv"
        error_values[i] = np.genfromtxt(filename, delimiter=",")
    mean_values = [
        np.mean(error_values[:, 0]),
        np.mean(error_values[:, 1]),
        np.mean(error_values[:, 2]),
    ]
    std_values = [
        np.std(error_values[:, 0]),
        np.std(error_values[:, 1]),
        np.std(error_values[:, 2]),
    ]

    # average ensembling, takes the mean of the predictions
    for i in range(0, len(job_parameters["ensemble_list"])):
        filename = job_name + str(i) + "_test_outputs.csv"
        test_out = np.genfromtxt(filename, delimiter=",", skip_header=1)
        if i == 0:
            test_total = test_out
        elif i > 0:
            test_total = np.column_stack((test_total, test_out[:, 2]))

    ensemble_test = np.mean(np.array(test_total[:, 2:]).astype(np.float), axis=1)
    ensemble_test_error = getattr(F, training_parameters["loss"])(
        torch.tensor(ensemble_test),
        torch.tensor(test_total[:, 1].astype(np.float)),
    )
    test_total = np.column_stack((test_total, ensemble_test))
    
    ##Print performance
    for i in range(0, len(job_parameters["ensemble_list"])):
        print(
            job_parameters["ensemble_list"][i]
            + " Test Error: {:.5f}".format(error_values[i, 2])
        )
    print(
        "Test Error Avg: {:.3f}, Test Standard Dev: {:.3f}".format(
            mean_values[2], std_values[2]
        )
    )
    print("Ensemble Error: {:.5f}".format(ensemble_test_error))
    
    ##Write output
    if write_output == "True" or write_output == "Partial":
        with open(
            str(job_name) + "_test_ensemble_outputs.csv", "w"
        ) as f:
            csvwriter = csv.writer(f)
            for i in range(0, len(test_total) + 1):
                if i == 0:
                    csvwriter.writerow(
                        [
                            "ids",
                            "target",
                        ]
                        + job_parameters["ensemble_list"]
                        + ["ensemble"]
                    )
                elif i > 0:
                    csvwriter.writerow(test_total[i - 1, :])
    if write_output == "False" or write_output == "Partial":
        for i in range(0, len(job_parameters["ensemble_list"])):
            filename = job_name + str(i) + "_errorvalues.csv"
            os.remove(filename)
            filename = job_name + str(i) + "_test_outputs.csv"
            os.remove(filename)

##Obtains features from graph in a trained model and analysis with tsne
def analysis(
    dataset,
    model_path,
    tsne_args,
):

    # imports
    #from sklearn.decomposition import PCA
    #from sklearn.manifold import TSNE
    #import matplotlib.pyplot as plt
    
    # imports
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from torch_geometric.data import Data
    import json
    
    rank = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    save_path = 'plots/'
    filename_prefix = 'plot_'
    
    dictionary_file = 'atom_init.json'
    with open(dictionary_file,'r') as json_file:
        data_dict = json.load(json_file)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    
    color_dic = {
                    '1':'gray',
                    '6':'yellow',
                    '7':'darkblue',
                    '8':'lightblue',
                
                }
                
    size_dic = {
                    '1':1,
                    '6':6,
                    '7':7,
                    '8':8,
                
                }
                
    def get_value(key):
        return color_dic.get(key, 'magenta')
    def get_size(key):
        return size_dic.get(key, 15)
    
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,#False,
        num_workers=0,
        pin_memory=True,
    )
    
    assert os.path.exists(model_path), "saved model not found"
    if str(rank) == "cpu":
        saved = torch.load(model_path, map_location=torch.device("cpu"))
    else:
        saved = torch.load(model_path, map_location=torch.device("cuda:1"))
    
    model = saved["full_model"]
    model = model.to(rank)
    model_summary(model)
    
    ##Get predictions
    time_start = time.time()
   
    model.eval()
    count = 0
    
    
    #fig = plt.figure()
    for data in loader:
         
        #print(count)
        data = data.to('cuda:1')
        with torch.no_grad():
            edge_index_1, edge_index_2, edge_index_3, score_1,score_2,score_3, edge_index, positions, batch, parameters, structure_id, x = model(data)
           #print(edge_index_1.shape, edge_index_2.shape, edge_index_3.shape, score_1.shape,score_2.shape,score_3.shape, edge_index.shape, positions.shape, batch.shape, parameters.shape,x.shape)
           #print(edge_index_3)
            unique,count_items = torch.unique(batch, return_counts=True)
            
            item_sum = torch.zeros(128)
            data_list = []
            ranges = []
            
            for num in range(128):
                
                if num ==0:
                    item_sum[num]=count_items[num]
                    index_range = (0,item_sum[num])
                    ranges.append(index_range)
                    #data = Data()
                    #data.edge_index_1 = edge_index_1[:,0:]
                else:
                    item_sum[num]=item_sum[num-1]+count_items[num]
                    index_range = (item_sum[num-1],item_sum[num])
                    ranges.append(index_range)
            
            for idss, (lower, upper) in enumerate(ranges):
                data = Data()
                classified_index_1_0 = edge_index_1[0][(edge_index_1[0] >= lower)&(edge_index_1[0] < upper)]
                classified_index_1_1 = edge_index_1[1][(edge_index_1[1] >= lower)&(edge_index_1[1] < upper)]
                classified_index_1 = torch.cat([classified_index_1_0.unsqueeze(0), classified_index_1_1.unsqueeze(0)], 0) 
                
                classified_index_2_0 = edge_index_2[0][(edge_index_2[0] >= lower)&(edge_index_2[0] < upper)]
                classified_index_2_1 = edge_index_2[1][(edge_index_2[1] >= lower)&(edge_index_2[1] < upper)]
                classified_index_2 = torch.cat([classified_index_2_0.unsqueeze(0), classified_index_2_1.unsqueeze(0)], 0) 
                
                classified_index_3_0 = edge_index_3[0][(edge_index_3[0] >= lower)&(edge_index_3[0] < upper)]
                classified_index_3_1 = edge_index_3[1][(edge_index_3[1] >= lower)&(edge_index_3[1] < upper)]
                classified_index_3 = torch.cat([classified_index_3_0.unsqueeze(0), classified_index_3_1.unsqueeze(0)], 0) 
               #print(classified_index_3)
                
                if idss == 0:
                    start_index_1 = 0
                    start_index_2 = 0
                    start_index_3 = 0
                    
                    start_position_index = 0
                else:
                    start_index_1 = end_index_1
                    start_index_2 = end_index_2
                    start_index_3 = end_index_3
                    
                    start_position_index = end_position_index
                    
                end_index_1 = start_index_1 + classified_index_1.shape[1]
                end_index_2 = start_index_2 + classified_index_2.shape[1]
                end_index_3 = start_index_3 + classified_index_3.shape[1]
                end_position_index = start_position_index + count_items[idss]
                
                node_positions = positions[start_position_index:end_position_index,:]
                node_x = x[start_position_index:end_position_index,:]
                
                mof_id = structure_id[idss]
                
                pressure = int(parameters[idss][0]*100)
                temprature = torch.round(parameters[idss][1].to(torch.float32))
                
                #print(count_items[idss])
                #print(node_positions.shape)
                #print(start_index_2)
                #print(end_index_2)
                attention_score_1 = score_1[start_index_1:end_index_1]
                #start_index_1 = end_index_1
                
                #end_index_2 = classified_index_2.shape[1]
                attention_score_2 = score_2[start_index_2:end_index_2]
                #start_index_2 = end_index_2
                #print(classified_index_2.shape)
                #print(attention_score_2.shape)
                
                #end_index_3 = classified_index_3.shape[1]
                attention_score_3 = score_3[start_index_3:end_index_3]
                #start_index_3 = end_index_3
                
                matching_atoms = []
                for node_x_item in node_x:
                    node_x_item = node_x_item.tolist()
                    node_x_item = [round(item) for item in node_x_item]
                   # print(node_x_item)
                    for key, value in data_dict.items():
                       # print(value)
                        
                        if value == node_x_item:
                            #print(key)
                            matching_atoms.append(key)
                            break
                if not matching_atoms:
                    print('did not find matching atoms!')
                #else:
                #    print(matching_atoms[0])
                #matching_atoms = np.array(matching_atoms).reshape(-1,1)
                
                data.atoms = matching_atoms
                data.pos = positions#node_positions
                data.nodes = node_positions
                data.edge_index_1 = classified_index_1
                data.edge_index_2 = classified_index_2
                data.edge_index_3 = classified_index_3
                data.edge_attr_1 = attention_score_1
                data.edge_attr_2 = attention_score_2
                data.edge_attr_3 = attention_score_3
                
                #print(data)
                #print(data.
                
                fig = plt.figure(dpi=1024)
                fig.suptitle(f'{mof_id} at 0.{pressure}bar {temprature}K',fontsize = 16)
                
                
                ax1 = fig.add_subplot(131,projection='3d')
                
                
                #print(get_value(matching_atoms[0]))
                #draw nodes
                for node_i, atom_i in zip(data.nodes, data.atoms):
                    color = get_value(atom_i)
                    size = get_size(atom_i)
                    ax1.scatter(node_i[0].cpu(), node_i[1].cpu(), node_i[2].cpu(), s = size, c = color )
                    
                
                #draw edges
                for src_1, dst_1, alpha_1 in zip(data.edge_index_1[0], data.edge_index_1[1], data.edge_attr_1):
                    #print(alpha)
                    score_alpha_1 = (alpha_1-data.edge_attr_1.min())/(data.edge_attr_1.max()-data.edge_attr_1.min())
                    #print(score_alpha_1.item())
                    ax1.plot([data.pos[src_1,0].cpu(), data.pos[dst_1,0].cpu()],
                             [data.pos[src_1,1].cpu(), data.pos[dst_1,1].cpu()],
                             [data.pos[src_1,2].cpu(), data.pos[dst_1,2].cpu()],
                             color='red', alpha = score_alpha_1.item(), lw=1)
               
                
                ax1.set_title('0~3$\AA$')
                
                ax2 = fig.add_subplot(132,projection='3d')
                #draw nodes
                for node_i, atom_i in zip(data.nodes, data.atoms):
                    color = get_value(atom_i)
                    size = get_size(atom_i)
                    ax2.scatter(node_i[0].cpu(), node_i[1].cpu(), node_i[2].cpu(), s = size, c = color )
                
                #draw edges
                for src_2, dst_2, alpha_2 in zip(data.edge_index_2[0], data.edge_index_2[1], data.edge_attr_2):
                    #print(alpha)
                    score_alpha_2 = (alpha_2-data.edge_attr_2.min())/(data.edge_attr_2.max()-data.edge_attr_2.min())
                    #print(score_alpha_2.item())
                    ax2.plot([data.pos[src_2,0].cpu(), data.pos[dst_2,0].cpu()],
                             [data.pos[src_2,1].cpu(), data.pos[dst_2,1].cpu()],
                             [data.pos[src_2,2].cpu(), data.pos[dst_2,2].cpu()],
                             color='green', alpha = score_alpha_2.item(), lw=1)
                
                ax2.set_title('3~8$\AA$')
                
                
                
                ax3 = fig.add_subplot(133,projection='3d')
                
                
                #draw edges
                for src_3, dst_3, alpha_3 in zip(data.edge_index_3[0], data.edge_index_3[1], data.edge_attr_3):
                   #if idss ==0:
                   #    print(idss)
                   #    print(data.nodes[int(src_3)-int(item_sum[idss])])
                   #    print(data.pos[src_3])
                  # else:
                  #     print(idss)
                  #     print(data.nodes[int(src_3)-int(item_sum[idss-1])])
                  #     print(data.pos[src_3])
                    #print(alpha_3)
                    score_alpha_3 = (alpha_3-data.edge_attr_3.min())/(data.edge_attr_3.max()-data.edge_attr_3.min())
                    #print(score_alpha_2.item())
                   #print(data.pos[src_3])
                   #print(data.pos[dst_3])
                    
                    ax3.plot([data.pos[src_3,0].cpu(), data.pos[dst_3,0].cpu()],
                             [data.pos[src_3,1].cpu(), data.pos[dst_3,1].cpu()],
                             [data.pos[src_3,2].cpu(), data.pos[dst_3,2].cpu()],
                             color='black', alpha = score_alpha_3.item(), lw=3)
                
                #draw nodes
                for node_i, atom_i in zip(data.nodes, data.atoms):
                    color = get_value(atom_i)
                    size = get_size(atom_i)
                    ax3.scatter(node_i[0].cpu(), node_i[1].cpu(), node_i[2].cpu(), s = size, c = color )
                
                ax3.set_title('8~15$\AA$')
                
                
                save_filename = os.path.join(save_path, filename_prefix + f'{mof_id}_0.{pressure}bar_{temprature}K.png')
                plt.savefig(save_filename,dpi=1024)
                
                #
                #plt.show()
                
                plt.close()
    
    
    
    
        count = count+1
    elapsed_time = time.time() - time_start
    
    print("Evaluation time (s): {:.5f}".format(elapsed_time))
    

    


