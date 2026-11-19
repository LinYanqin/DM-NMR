#!/bin/csh -f


echo '|   Processing time domain MDD reconstruction '
echo
echo Processing YZ dimensions
xyz2pipe -in data/test%03d.fid -x -verb              \
| nmrPipe  -fn POLY -time                           \
| nmrPipe  -fn SP -size 512 -off 0.5 -end 1 -pow 2 -c 0.5  \
| nmrPipe  -fn ZF -auto                             \
| nmrPipe  -fn FT                                   \
| nmrPipe  -fn PS -p0 0 -p1 0.0 -di              \
| nmrPipe  -fn EXT -x1 10ppm -xn 6.2ppm -sw -verb \
| nmrPipe  -fn POLY -auto -ord 0                    \
| nmrPipe  -fn TP \
| nmrPipe  -fn FT                                   \
| nmrPipe  -fn PS -p0  -93 -p1  0               \
| nmrPipe  -fn FT    -inv       \
| nmrPipe  -fn SP -off 0.50 -end 1 -pow 2 -c 0.5  \
| nmrPipe  -fn ZTP \
| nmrPipe  -fn SP -off 0.50 -end 1 -pow 2 -c 0.5  \
| nmrPipe  -fn PS -p0  0 -p1  0                 \
| nmrPipe  -fn ZTP \
| nmrPipe  -fn TP \
  > ./fid_temp.dat
exit
