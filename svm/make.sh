#!/bin/bash

TARGET=main
FILES=`ls | egrep "\.h|\.cpp"`
echo $FILES
g++ ${FILES} -o ${TARGET}
