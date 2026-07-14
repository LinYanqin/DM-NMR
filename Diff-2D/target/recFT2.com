#!/bin/csh -f


echo '|   Processing time domain MDD reconstruction '
echo
echo Processing YZ dimension
set I = 1
echo $I
set FID_file = ./fid_temp.dat
xyz2pipe -in $FID_file            \
| nmrPipe  -fn TP -auto           \
| nmrPipe  -fn ZF -size 128        \
| nmrPipe  -fn SP -off 0.5 -end 1 -pow 2 -c 0.5 \
| nmrPipe  -fn FT          \
| nmrPipe  -fn PS -p0 0.0 -p1 -0.0  -di\
| nmrPipe  -fn TP  -auto                            \
| nmrPipe  -fn ZTP                                  \
| nmrPipe  -fn ZF -size 128	          \
| nmrPipe  -fn SP -off 0.5 -end 1 -pow 2 -c 0.5 \
| nmrPipe  -fn FT -alt                   \
| nmrPipe  -fn PS -p0  0.0 -p1 0 -di\
| nmrPipe  -fn ZTP      \
| nmrPipe  -fn TP  -auto                            \
| pipe2xyz -out ./fidtemp3D.ft3 -x -verb -ov
	


