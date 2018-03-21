# Project: squeezeDetOnKeras
# Filename: scheduler
# Author: Christopher Ehmann
# Date: 09.01.18
# Organisation: searchInk
# Email: christopher@searchink.com



import sys
import subprocess
import argparse
from datetime import datetime
import os



def run_schedule(train_script="train.py", eval_script="eval.py", schedule="schedule.config", no_eval=0):

    """Runs a schedule file. Check the example config file. Every flag should be named as the argument that is passed, except img and gt
    for training, which should be called train_img and train_gt.
    
    Keyword Arguments:
        train_script {str} -- path to train.py script (default: {"train.py"})
        eval_script {str} -- path to eval.py script (default: {"eval.py"})
        schedule {str} -- schedule to run (default: {"schedule.config"})
        no_eval {int} -- If you just want to run trainings (default: {0})
    """

    defaults_train = {}

    defaults_eval = {}

    #read config
    with open(schedule) as f:
        for line in f:
            #skip comments
            if not line.startswith("#"):
            #check for header
                if line.startswith('%'):
                    args = line[1:-1].split(" ")
                    #get default arguments
                    for a in args:
                        #skip empties
                        if not a  == "":
                            arg, val = a.split(":")

                            #set evaluation and training images and gt accordingly
                            if "val" in arg or "test" in arg:
                                if "gpu" in arg:
                                    defaults_eval[ arg.split("_")[1] ] = val
                                else:
                                    defaults_eval[arg] = val


                            elif "train" in arg:
                                defaults_train[arg.split("_")[1]] = val
                            elif "experiment" in arg:
                                experiment = val
                            elif "epochs" in arg or "steps" in arg or "logdir" in arg:
                                defaults_eval[arg] = val
                                defaults_train[arg] = val
                            else:
                                defaults_train[arg] = val


                #actual run
                else:

                    #skip empty lines
                    if not line.strip() == "":
                        args_train = defaults_train
                        args_eval = defaults_eval
                        for a in line.strip().split(" "):
                            if not a == "":
                                arg, val = a.split(":")
                                # set evaluation and training images and gt accordingly

                                # set evaluation and training images and gt accordingly
                                if "val" in arg or "test" in arg:
                                    defaults_eval[arg] = val
                                elif "train" in arg:
                                    defaults_train[arg.split("_")[1]] = val
                                elif "experiment" in arg:
                                    experiment = val
                                elif "epochs" in arg or "steps" in arg or "logdir" in arg:
                                    print(arg)
                                    defaults_eval[arg] = val
                                    defaults_train[arg] = val
                                else:
                                    defaults_train[arg] = val


                        #create the commands for running the training script and the eval
                        train_command ="python " +  train_script


                        has_log_dir = False
                        GPUS = 1
                        logdir = None

                        for arg, val in args_train.items():


                            #check if logdir is given
                            if arg == "logdir":
                                has_log_dir = True
                                logdir = val

                                train_command += " --" + arg + " " + experiment +"/" + val

                            #if multigpu, try to use last gpu for evaluation if not specified otherwise
                            elif arg == "gpus":
                                GPUS = int(val)
                                train_command += " --" + arg + " " + val

                            else:
                                train_command += " --" + arg + " " + val


                        eval_command = "python " + eval_script


                        for arg, val in args_eval.items():
                            # check if logdir is given

                            if arg == "logdir":
                                has_log_dir = True
                                logdir = val
                                eval_command += " --" + arg + " " + experiment + "/" + val

                            else:
                                eval_command += " --" + arg + " " + val


                        #when no gpus for evaluation where set, use the last one on the machine
                        if GPUS > 1:

                            eval_command += " --gpu " + str(GPUS)

                        #create a custom name for logging directory if none is given
                        if not has_log_dir:
                            logdir = experiment + "/" + datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                            train_command += " --logdir " + logdir
                            eval_command += " --logdir " + logdir

                        else:
                            logdir = experiment + "/" + logdir



                        #create dirs
                        if not os.path.exists(logdir):
                            os.makedirs(logdir)



                        #write the output into the log dirs
                        training_output = open( logdir + "/train.out", "w+")
                        eval_output = open( logdir + "/eval.out", "w+")

                        print("------------------")
                        print("Running ")
                        print(train_command)

                        if not no_eval:
                            print("and ")
                            print(eval_command)

                        training_process = subprocess.Popen(train_command, stdout=subprocess.PIPE, stderr= subprocess.PIPE, shell=True)

                        if not no_eval:
                            eval_process = subprocess.Popen(eval_command, stdout=subprocess.PIPE, stderr= subprocess.PIPE, shell=True)

                        output, err =  training_process.communicate()

                        #print(err)
                        if not no_eval:

                            output_eval, err_eval = eval_process.communicate()
                            #print(err_eval)

                        #wait for training to end
                        training_process.wait()
                        training_output.write(err.decode('utf-8'))
                        training_output.write(output.decode('utf-8'))
                        training_output.close()
                        if not no_eval:

                            eval_process.wait()
                            eval_output.write(err_eval.decode('utf-8'))
                            eval_output.write(output_eval.decode('utf-8'))
                            eval_output.close()

    print("Schedule completed")








if __name__ == "__main__":

    train_script = "train.py"
    eval_script = "eval.py"
    schedule = "schedule.config"
    experiment = "experiments_of_" + datetime.now().strftime('%Y-%m-%d-%H:%M')
    no_eval = 0

    parser = argparse.ArgumentParser(description='Run a sequence of squeezedet trainings')
    parser.add_argument("--train", help="path to training script. DEFAULT: train.py", default = "train.py")
    parser.add_argument("--eval", help="path to evaluation script. DEFAULT: eval.py", default ="eval.py")
    parser.add_argument("--no_eval", type= bool, help="Run evaluation DEFAULT: False", default = 0)
    parser.add_argument("--schedule", help="path to training schedule. DEFAULT: schedule.config", default="schedule.config")

    args = parser.parse_args()

    if args.train is not None:
        train_script = args.train
    if args.eval is not None:
        eval_script = args.eval
    if args.schedule is not None:
        schedule = args.schedule
    if args.no_eval is not None:
        no_eval = args.no_eval

    run_schedule(train_script, eval_script, schedule, no_eval)



