#!/bin/bash
x=${2-1}
addr=$1
while read -r line
do
    echo "iteration: $x"
    bash -c "$line" || echo "done"
    echo "done"
    echo "the line is:"
    echo $line
    echo ""
    echo ""
    x=$(echo "$x + 1" | bc)
    echo "incr x to: $x"
    sleep 3
done < <(tail -n +$x "./runs/$addr/run.sh")
