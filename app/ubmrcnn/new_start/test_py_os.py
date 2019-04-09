import os,sys,argparse,logging,time
import getpass

parser = argparse.ArgumentParser()
parser.add_argument("-d","--device",type=str,default="1",
                    help="batch size for each worker")

args = parser.parse_args(sys.argv[1:])
os.environ["CUDA_VISIBLE_DEVICES"]=args.device
is_in = "CUDA_VISIBLE_DEVICES" in os.environ
print(is_in)
if is_in:
    print(os.environ["CUDA_VISIBLE_DEVICES"])

# for k,v in os.environ.items():
#     print(k, v)
