#!/bin/csh

nmrPipe -in ./test.fid  \
| nmrPipe  -fn SOL \
| nmrPipe  -fn SP -off 0.5 -end 0.95 -pow 2  -c 0.5  \
| nmrPipe  -fn FT -auto                                 \
| nmrPipe  -fn PS -p0 137 -p1 0.0 -di -verb       \
| nmrPipe  -fn EXT -x1 12.0ppm -xn 6.0ppm -sw \
| nmrPipe  -fn TP                                   \
| nmrPipe  -fn SP -off 0.5 -end 0.99 -pow 1  -c 0.5  \
| nmrPipe  -fn PS -p0 0 -p1 0               \
   -verb -ov -out test.ft1

nmrPipe -in ./test.ft1  \
| nmrPipe  -fn FT                               \
   -verb -ov -out test.ft2
