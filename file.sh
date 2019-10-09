#!/bin/bash

wait_for_string() {
    target=$1
    shift
    ( tail -f $@ & ) | grep -q $target
}

wait_for_string hello a b
