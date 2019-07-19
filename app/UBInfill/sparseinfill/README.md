Below are notes on how to the run scripts (and where)

## Run local client/worker/broker combo in one job

Can do this using one of the flags in the client script: `start_ublarcvserver_sparseinfill_client.py`

      ./start_ublarcvserver_sparseinfill_client.py -brokeraddr ipc:///tmp/infill/test -l test.log -d -i ../../../../testdata/mcc9v12_intrinsicoverlay/supera-Run004955-SubRun000079.root -a wire -c wire -o test.root --local --weights-dir weights/prelim_june2019/

Note: Make sure the IPC socket name ('/tmp/infill/test') is unique (on the machine).
Also, one must make the directory for the socket name (which under the hood is a file-like object, I believe).
So in the example, one needs to make the directory `/tmp/infill` first, before running the client.
If a job runs the same machine, you are at risk of sending client events to the wrong broker/worker.  
