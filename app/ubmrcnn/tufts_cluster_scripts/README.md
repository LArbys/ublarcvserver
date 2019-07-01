# Tufts Cluster Scripts

Folder to contain the scripts to launch UB-MRCNN Workers on the Tufts Cluster.

We assume we work in a singularity container.

Container recipe: [to do]

## Build `ubdl` for python3 on Tufts

If you want to run code based on your own copy, follow these instructions
to compile the depencies.  
You will want to do this if you are planning to run a development copy.

## Start UBRMCN Worker

To start a worker:

     sbatch submit_ubmrcnn_worker_pgpu03.sh


