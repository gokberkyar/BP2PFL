# P2PML

## How to run ?

    There are three ways to run experiments: single interactive, single background, multiple automated

### How to run single experiment in an interactive way?

```
 	# activate environment and cd to project folder, may use venv command in bashrc setup
 	# you should be inside project folder p2pml 
 	# there must be Experiments folder in the current dir
 	# experiment_'exp_no'.json should exist inside the Experiments folder
 	p2pml 'exp_no' 
 	# this command should automaticly create docker containers
 	# can verify with docker ps command within few mins 
 	# you expect to see worker0 ... worker60 etc.

        # alternatively, you can run the main file directly:
        python3 -u p2pml/src/main.py 'exp_no'

```

### How to run single experiment in background ?

    1. Complete Bashrc setup
    2. run 'exp_no' 'current_machine_id'

    Alternatively, you can run the following command in the command prompt:
    nohup python3 -u p2pml/src/main.py 'exp_no' &> out.txt &

### Useful docker commands

    docker ps    => list all running peers
    docker logs peer0  => visualize the logs of peer0
    docker stop $(docker ps -q) => to stop them all gracefully

#### Bashrc Setup

    1. Copy paste below commands to your bashrc
    2. venv: automaticly enables your virtualenv and cd to project directory
    3. clean_exp: Stops all docker containers but doesn't removes them so you can still query docker logs peer0.
    4. run: inputs exp_num and machine_id as input, runs the experiment in background and creates a log file nohup_'expno'_'machine_no'.txt, be aware it doesnt automate to ssh to machine, machine_no just for logging purpose.`` 	alias venv="source /net/data/p2pml/env/bin/activate && cd p2pml" 	clean_exp(){    	 command="docker container stop $(docker container ls -q --filter name=peer*)"     	eval $command 	} 	run(){    	# t stores $1 argument passed to fresh()    	exp=$1    	machine=$2    	venv="source /net/data/p2pml/env/bin/activate && cd p2pml"       	command="PYTHONUNBUFFERED=1 $venv && nohup p2pml $exp > nohup$exp-$machine.txt 2> nohup$exp-$machine.txt &"    	echo $command    	eval $command 	} 	``

## How to create Experiments folder?

```
	venv
	p2pml-cli experiment all
```

## How to create id.json from strach?

```shell
    cd p2pml/Networking/peerId
    docker build -t peeridgen .
    docker run peeridgen > id.json
```
