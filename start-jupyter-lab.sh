#!/bin/sh

HOST_IP=`/sbin/ip route get 8.8.8.8 | awk '{print $7;exit}'`

jupyter-lab --no-browser --ip=$HOST_IP --port=15021
