#!/bin/bash

export MACHINE=$1
export NODE_NUM=${MACHINE:1}
export CMD=$2

FELISCTL_OUT=felis-ctl.out
FELISCTL_ERR=felis-ctl.err
FELIS_OUT=felis.out
FELIS_ERR=felis.err

FELISCTL_CFG=$(cat <<EOF
{
  "type": "status_change",
  "status": "configuring",
  "mem": {
  },
  "nodes": [
    {
      "name": "host1",
      "ssh_hostname": "c$NODE_NUM",
      "worker": {"host": "142.150.234.$NODE_NUM", "port": 1091},
      "index_shipper": {"host": "142.150.234.$NODE_NUM", "port": 43411}
    }
  ],
  "tpcc": {
    "hot_warehouses": [],
    "offload_nodes": []
  },
  "controller": {
    "http_port": 8666,
    "rpc_port": 3148
  }
}
EOF
)

WORKSPACE_DIR=workspace
FELIS_DIR=workspace/felis
FELISCTL_DIR=workspace/felis-controller

wait_for_string() {
    target="$1"
    shift
    ( tail -f $@ & ) | grep -q "$target"
}

run_felis_controller() {
    then_dir=$(pwd)
    cd $FELISCTL_DIR

    printf "%s" "$FELISCTL_CFG" >temp-cfg.json.template
    envsubst '$NODE_NUM' <temp-cfg.json.template > temp-cfg.json

    $MILL FelisController.run temp-cfg.json 1>$TEMP_DIR/$FELISCTL_OUT 2>$TEMP_DIR/$FELISCTL_ERR &
    
    echo "waiting for felis-controller"
    wait_for_string 'Server all initialized.' $TEMP_DIR/$FELISCTL_OUT $TEMP_DIR/$FELISCTL_ERR
    echo "felis-controller up"

    cd "$then_dir"
}

run_felis() {
    then_dir=$(pwd)
    cd $FELIS_DIR
    echo "running: '$@'"
    $@ 1>$TEMP_DIR/$FELIS_OUT 2>$TEMP_DIR/$FELIS_ERR &
    echo "$@" > "$FINAL_DIR/cmd"
    printf "%s" "$meta" > "$FINAL_DIR/meta"

    echo "waiting for felis"
    wait_for_string 'Ready. Waiting for run command from the controller.' $TEMP_DIR/$FELIS_OUT $TEMP_DIR/$FELIS_ERR
    echo "felis up"

    cd "$then_dir"
}

wait_to_finish() {
    wait_for_string 'Freeing EpochMemory' $TEMP_DIR/$FELIS_OUT $TEMP_DIR/$FELIS_ERR
    wait_for_string 'disconnected' $TEMP_DIR/$FELISCTL_OUT $TEMP_DIR/$FELISCTL_ERR
}

if [[ $remote ]] ; then
    ps aux | grep felis | grep contr | awk '{ print $2 }' | xargs kill >/dev/null 2>&1 || echo
    ps aux | grep release | grep db | grep -v run\\-job | awk '{ print $2 }' | xargs kill >/dev/null 2>&1 || echo

    FINAL_DIR=$(realpath workspace/felis-benchmarks/skew-core-scaling/$MACHINE-$(date +%s))
    export LD_LIBRARY_PATH=/pkg/ld_lib
    export TEMP_DIR=$(mktemp -d)
    echo $TEMP_DIR
    echo $FINAL_DIR

    mkdir -p $FINAL_DIR

    export PATH=/pkg/java/j9/bin:$PATH
    export JAVA_HOME=/pkg/java/j9
    export MILL=/mnt/home/pkg/pkg/bin/mill

    echo "in remote: '$@'"
    while [[ $(echo "$(mpstat | egrep -o [0-9]+.?[0-9]*$) < 99" | bc) -eq 1 ]] ; do
        echo "Node too busy..."
        sleep 0.5
    done
    
    run_felis_controller
    shift
    run_felis $@
    curl localhost:8666/broadcast/ -d '{"type": "status_change", "status": "connecting"}'
    wait_to_finish

    cp $TEMP_DIR/* $FINAL_DIR/

    ps aux | grep felis | grep contr | awk '{ print $2 }' | xargs kill >/dev/null 2>&1 || echo
    ps aux | grep tail | grep farnasi | grep felis -i | awk '{ print $2 }' | xargs kill >/dev/null 2>&1 || echo
fi

if [[ $local ]] ; then
    printf "%s" "$meta"
    scp $0 $MACHINE:$WORKSPACE_DIR >/dev/null 2>&1
    /usr/bin/ssh -n $MACHINE meta="'""${meta}""'" remote=1 ./$WORKSPACE_DIR/$0 $@
    echo ""
    echo ""
fi
