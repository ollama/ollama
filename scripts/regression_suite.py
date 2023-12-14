import subprocess
import sys
import time
from binaries import Binaries
from machine import Machine
from paramiko import SSHClient
import logging
import yaml
import os

# Very rudimentary regression suite to execute hello world remotely and gather some results
# TODO - this should be revampted into some test framework to be able to parallelize

TEST_CONFIG="./test-systems.yaml"

if __name__ == '__main__':
    if not os.path.exists(TEST_CONFIG):
        print("ERROR: you must create a test config file")
        sys.exit(1)
    with open(TEST_CONFIG) as cfgFile:
        cfg = yaml.safe_load(cfgFile)

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    # Dumb algorithm for now, just serial
    for lib in cfg:
        # print("Testing type " + lib)
        for t in cfg[lib]:
            tgt = cfg[lib][t]
            variant = tgt.get("variant", "")
            port = tgt.get("port")
            conn = tgt.get("connect", t)
            log.info("Testing on " + t + " " + variant + " connecting via " + conn)
      
            ssh = SSHClient()
            ssh.load_system_host_keys()
            m = Machine(ssh, conn, port=port, variant=variant)
            m.assesMachine()
            b = Binaries("./dist")
            m.copyBinary(b, False, True)
            m.startServer()
            time.sleep(5)
            data = m.runClientOneShot("run orca-mini hello")
            # TODO - we could verify the data has something we expect...
            sys.stdout.buffer.write(data)
            sys.stdout.flush()
            m.stopServer()

    # # Now aggregate the coverage results and pop up a report
    # BUSTED - 
    # go tool cover -html ./out.txt
    # cover: inconsistent NumStmt: changed from 4 to 3
    # log.info("Generating coverage report...")
    # subprocess.run("go tool covdata textfmt -i=coverage -o coverage/cov.txt", shell=True, check=True)
    # subprocess.run("go tool cover -html ./coverage/cov.txt", shell=True, check=True)


# After running tests across multiple systems view the coverage results with:
#
# go tool covdata textfmt -i=coverage -o coverage/cov.txt
# go tool cover -html ./coverage/cov.txt 